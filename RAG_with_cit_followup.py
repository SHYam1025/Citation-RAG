import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv() 

# LangChain components
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser # For parsing the forced JSON output

# Check API key
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    print("WARNING: HUGGINGFACEHUB_API_TOKEN is not set.")
if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY is not set.")

# --- 1. Define Citation Schema (Pydantic) ---
class CitedAnswer(BaseModel):
    """The final answer and the specific source chunk IDs used to generate it."""
    answer: str = Field(..., description="The complete, direct answer to the user question, based ONLY on the provided context.")
    citations: List[int] = Field(..., description="The integer IDs of the SPECIFIC source chunks (chunk_id) which justify the answer.")

# --- 2. Configuration ---
# Hugging Face Model for Embeddings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Text Splitter Config (MUST be consistent across all loading/splitting)
SPLITTER_CONFIG = {
    "chunk_size": 900,
    "chunk_overlap": 120,
    "separators": ["\n\n", "\n", " ", ""],
    "add_start_index": True
}

# Global variables
VECTOR_STORE = None
ALL_CHUNKS_FOR_LOOKUP: List[Document] = []

# --- 3. RAG Pipeline Functions ---
def initialize_vector_store(pdf_path: str):
    """Loads PDF, splits it, and creates a FAISS vector store using HF Embeddings."""
    global VECTOR_STORE, ALL_CHUNKS_FOR_LOOKUP
    
    # 1. Load Documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 2. Split and Assign ID (Using global config)
    text_splitter = RecursiveCharacterTextSplitter(**SPLITTER_CONFIG)
    chunks = text_splitter.split_documents(documents)
    
    # Assign a unique, simple integer ID to each chunk
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
        
    # Store the chunks for the final citation lookup
    ALL_CHUNKS_FOR_LOOKUP = chunks
    
    # 3. Create Vector Store with Hugging Face Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME, 
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    VECTOR_STORE = FAISS.from_documents(chunks, embeddings)
    print(f"✅ Vector Store created with {len(chunks)} chunks using {EMBEDDING_MODEL_NAME}.")

def format_docs_with_id(docs: List[Document]) -> str:
    """Formats retrieved documents to include a citation Source ID and content."""
    formatted = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        # Assuming page is a 0-based index; convert to 1-based for human readability
        # Handle cases where 'page' might not be an integer (e.g., "N/A")
        page_index = doc.metadata.get("page", -1)
        page_num = int(page_index) + 1 if isinstance(page_index, int) and page_index >= 0 else "N/A"
        # Ensure 'source' is in metadata; otherwise, default
        source = os.path.basename(doc.metadata.get("source", "PDF_Source"))
        
        # Format for the LLM prompt
        formatted.append(
            f"Source ID: {chunk_id}\n"
            f"Source: {source} (Page {page_num})\n"
            f"Content: {doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)

# In your create_rag_chain function, modify the prompt construction:
# In your create_rag_chain function, modify the prompt construction:
def create_rag_chain():
    """Constructs the LangChain RAG chain."""
    if VECTOR_STORE is None:
        raise ValueError("Vector store not initialized. Run initialize_vector_store first.")
        
    retriever = VECTOR_STORE.as_retriever(k=10)
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    
    output_parser = JsonOutputParser(pydantic_object=CitedAnswer)
    
    # 1. Get the format instructions from the parser
    # 2. ESCAPE the curly braces by replacing '{' with '{{' and '}' with '}}'.
    # This tells the ChatPromptTemplate to treat them as literal strings.
    escaped_format_instructions = output_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    
    # --- MODIFIED SYSTEM PROMPT ---
    system_prompt_text = (
        "You are a helpful and expert research assistant. "
        "Your task is to answer a user's question using only the provided documents. "
        "Do not include any information from your pre-existing knowledge base.\n\n"
        
        "Your response must be a single, structured JSON object that conforms exactly to the following format. "
        "Your final output should be ONLY the JSON object, with no extra text or conversational filler.\n\n"
        
        # Include the escaped format instructions.
        f"JSON OUTPUT FORMAT:\n{escaped_format_instructions}\n\n"
        
        "The 'citations' array must contain the integer 'Source ID' (chunk_id) "
        "of every specific source chunk that justifies your 'answer'.\n"
        "If the answer is not available in the provided documents, the 'answer' field should be 'Answer not available in the provided document,' and the 'citations' array should be empty.\n\n"
        
        "Context:\n{context}\n\n"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("user", "Question: {question}"),
        ]
    )
    
    # LCEL Chain (No change here)
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs_with_id), 
         "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | output_parser
    )
    return rag_chain

def generate_followups(question: str, answer: str, llm) -> List[str]:
    """Generate 3 natural follow-up questions based on the Q&A."""
    followup_prompt = (
        "You are an assistant that suggests 3 natural follow-up questions "
        "a user might ask after getting an answer. Format the output as a numbered list.\n\n"
        f"User's Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Generate 3 natural follow-up questions that build *directly* upon the information provided in the 'Answer' field, "
        "while remaining within the scope of the document. The questions should be specific and helpful for continued research. Start each with a number."
        "Start each with a number."
    )
    response = llm.invoke(followup_prompt)
    # Basic parsing to split the numbered list
    return [q.split(". ", 1)[-1].strip() for q in response.content.split("\n") if q.strip().startswith(('1.', '2.', '3.'))]

def query_pdf(question: str, rag_chain: Any, all_chunks: List[Document]) -> Dict[str, Any]:
    """Invokes the RAG chain and formats the final output."""
    
    # 1. Invoke the chain to get the structured answer
    try:
        structured_response = rag_chain.invoke(question)
        
        # Validation
        if not isinstance(structured_response, dict):
            # This is a fallback if the parser fails
            return {"answer": str(structured_response), "citations": [], "followups": []}
        
        # Manually convert dict to Pydantic object for type safety/validation if needed, 
        # but the parser should return the dict matching the model structure
        final_answer = structured_response.get('answer', 'Answer not available (parsing error).')
        cited_ids = structured_response.get('citations', [])
            
    except Exception as e:
        # Handle cases where the LLM returns completely unparseable text
        return {"answer": f"Error in structured response: {type(e).__name__}: {e}", "citations": [], "followups": []}
    
    # 2. Generate follow-up questions
    # The LLM from the chain is the second-to-last step (llm -> output_parser)
    followups = generate_followups(question, final_answer, rag_chain.steps[-2]) 
    
    # 3. Look up full citation details from the original chunks
    # Create the map from the globally stored chunks
    all_chunks_map = {doc.metadata['chunk_id']: doc.metadata for doc in all_chunks}
    
    unique_cited_ids = sorted(list(set(cited_ids)))
    
    formatted_citations = []
    for chunk_id in unique_cited_ids:
        metadata = all_chunks_map.get(chunk_id)
        if metadata:
            source = os.path.basename(metadata.get("source", "PDF"))
            # Apply the same 0-based to 1-based correction for final output lookup
            page_index = metadata.get("page", -1)
            page_num = int(page_index) + 1 if isinstance(page_index, int) and page_index >= 0 else "N/A"
            formatted_citations.append(f"{source} (Page {page_num}, Chunk ID: {chunk_id})")
            
    # 4. Return the formatted result
    return {
        "answer": final_answer,
        "citations": formatted_citations,
        "followups": followups
    }

# --- 4. Main Execution ---
if __name__ == "__main__":
    
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        print("ERROR: Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
        # exit() # Commenting out to allow execution if user wants to see other parts
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: Please set the GROQ_API_KEY environment variable.")
        # exit()
        
    # IMPORTANT: Replace this with the path to your PDF file
    PDF_FILE_PATH = "EOT-Al_Namaa_Poultry-May_2023.pdf" 
    
    if not os.path.exists(PDF_FILE_PATH):
        print(f"ERROR: PDF file not found at '{PDF_FILE_PATH}'.")
        print("Please place a PDF in the same directory and rename it to 'research_paper.pdf' (or update the path).")
        # exit() # Commenting out to allow execution
    
    try:
        # Step 1: Initialize the Vector Store (Load, Split, Embed, and store ALL_CHUNKS_FOR_LOOKUP)
        initialize_vector_store(PDF_FILE_PATH)
        
        # The ALL_CHUNKS_FOR_LOOKUP is now guaranteed to match the chunks in the vector store
        
        # Step 2: Create the RAG Chain
        rag_chain = create_rag_chain()
        
        # Step 3: Ask a Question
        user_question = input("\nEnter your question about the PDF: ")
        
        print("\n--- Generating Response via Groq/Hugging Face APIs ---\n")
        
        result = query_pdf(user_question, rag_chain, ALL_CHUNKS_FOR_LOOKUP)
        
        # Step 4: Display Results
        print("💡 **Answer**:")
        print(result['answer'])
        
        print("\n--- Citations (Source Details) ---")
        if result['citations']:
            for citation in sorted(list(set(result['citations']))):
                print(f"• {citation}")
        else:
            print("No relevant citations found in the document for this answer.")
            
        print("\n--- Suggested Follow-Up Questions ---")
        if result['followups']:
          for i, q in enumerate(result['followups'], 1):
           print(f"👉 {q}")
        else:
           print("No follow-up suggestions available.") 
           
        print("\n------------------------------\n")
        
    except ValueError as ve:
        print(f"\nConfiguration Error: {ve}")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {type(e).__name__}: {e}")
        print("Please check your API keys, model names, and network connection.")