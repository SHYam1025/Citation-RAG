import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# This line loads the variables from the .env file into os.environ
load_dotenv() 
from pydantic import BaseModel, Field

# LangChain components for Hugging Face
from langchain_huggingface import  HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
import os
from langchain import chains
from langchain_community.document_loaders import PyPDFLoader 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from langchain_core.output_parsers import JsonOutputParser
#from langchain.output_parsers import JsonOutputParser # For parsing the forced JSON output # For parsing the forced JSON output
# Now the key is accessible by LangChain and your code
print(os.getenv("HUGGINGFACEHUB_API_TOKEN")) 
# --- 1. Define Citation Schema (Pydantic) ---

class CitedAnswer(BaseModel):
    """The final answer and the specific sources used to generate it."""
    answer: str = Field(..., description="The complete, direct answer to the user question, based ONLY on the provided context.")
    citations: List[int] = Field(..., description="The integer IDs of the SPECIFIC source chunks which justify the answer.")

# --- 2. Configuration ---

# Hugging Face Model for Generation (Choose an instruction-tuned, strong model)
LLM_REPO_ID = "llama-3.3-70b-versatile"
# Hugging Face Model for Embeddings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Global variables (For simplicity, not persisted in this example)
VECTOR_STORE = None


# --- 3. RAG Pipeline Functions ---

def initialize_vector_store(pdf_path: str):
    """Loads PDF, splits it, and creates a FAISS vector store using HF Embeddings."""
    global VECTOR_STORE
    
    # 1. Load Documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split and Assign ID
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    
    # Assign a unique, simple integer ID to each chunk
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i

    # 3. Create Vector Store with Hugging Face Embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME, 
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN") # Note: parameter name often changes slightly
    )

    VECTOR_STORE = FAISS.from_documents(chunks, embeddings)
    print(f"✅ Vector Store created with {len(chunks)} chunks using {EMBEDDING_MODEL_NAME}.")


def format_docs_with_id(docs: List[Document]) -> str:
    """Formats retrieved documents to include a citation Source ID and content."""
    formatted = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        page_num = doc.metadata.get("page", "N/A")
        source = os.path.basename(doc.metadata.get("source", "PDF"))
        
        # Format for the LLM prompt
        formatted.append(
            f"Source ID: {chunk_id}\n"
            f"Source: {source} (Page {page_num})\n"
            f"Content: {doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def create_rag_chain():
    """Constructs the LangChain RAG chain using the Hugging Face Endpoint."""
    if VECTOR_STORE is None:
        raise ValueError("Vector store not initialized. Run initialize_vector_store first.")

    retriever = VECTOR_STORE.as_retriever(k=10)
    
    # Initialize HuggingFaceEndpoint for the LLM
    # Initialize ChatGroq for the LLM
    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=os.environ.get("GROQ_API_KEY") # You must set this env variable
)

    # We use a standard parser and heavily prompt the LLM to return JSON
    output_parser = JsonOutputParser(pydantic_object=CitedAnswer)
    
    # The prompt MUST explicitly tell the model to output a JSON object 
    # conforming to the required schema.
    format_instructions = output_parser.get_format_instructions()
    
    system_prompt = (
    "You are a helpful and expert research assistant. "
    "Your task is to answer a user's question using only the provided documents. "
    "Do not include any information from your pre-existing knowledge base.\n\n"
    
    "Your response must be a single, structured JSON object with two keys: `answer` and `citations`.\n\n"
    
    "The `answer` field must contain the full answer to the question.\n"
    "The `citations` field must be an array of objects. Each object must contain the `source_id`, `page`, and `chunk_id` for every piece of information used in the `answer` field.\n"
    "If the answer is not available in the provided documents, the `answer` field should be 'Answer not available in the provided document,' and the `citations` array should be empty.\n\n"
    
    "Your final output should be ONLY the JSON object, with no extra text or conversational filler.\n\n"
    
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Response (in JSON format):"
)
    

    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Context:\n\n{context}\n\nQuestion: {question}"),
        ]
    )

    # LCEL Chain
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
        "a user might ask after getting an answer.\n\n"
        f"User's Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Generate 3 follow-up questions that are related, curious, and helpful."
    )
    response = llm.invoke(followup_prompt)
    return [q.strip("-• ") for q in response.content.split("\n") if q.strip()]


def query_pdf(question: str, rag_chain: Any, all_chunks_for_lookup: List[Document]) -> Dict[str, Any]:
    """Invokes the RAG chain and formats the final output."""
    
    # 1. Invoke the chain to get the structured answer
    try:
        structured_response = rag_chain.invoke(question)
        
        # Validation: check dict vs Pydantic model
        if isinstance(structured_response, dict):
            final_answer = structured_response.get("answer", "")
            citations_raw = structured_response.get("citations", [])
            cited_ids = []
            for c in citations_raw:
                if isinstance(c, dict):
                    cited_ids.append(c.get("chunk_id", c.get("source_id")))
                elif isinstance(c, int):
                    cited_ids.append(c)
            cited_ids = [cid for cid in cited_ids if cid is not None]
        elif isinstance(structured_response, CitedAnswer):
            final_answer = structured_response.answer
            cited_ids = structured_response.citations
        else:
            print("Warning: Model output was malformed. Attempting best effort parsing.")
            return {"answer": str(structured_response), "citations": [], "followups": []}
            
    except Exception as e:
        # Handle cases where the LLM returns completely unparseable text
        return {"answer": f"Error in structured response: {e}", "citations": [], "followups": []}
    followups = generate_followups(question, final_answer, rag_chain.steps[-2]) 
    # 3. Look up full citation details from the original chunks
    all_chunks_map = {doc.metadata['chunk_id']: doc.metadata for doc in all_chunks_for_lookup}
    
    unique_cited_ids = sorted(list(set(cited_ids)))
    
    formatted_citations = []
    for chunk_id in unique_cited_ids:
        metadata = all_chunks_map.get(chunk_id)
        if metadata:
            source = os.path.basename(metadata.get("source", "PDF"))
            page_num = metadata.get("page", "N/A")
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
        exit()

    # IMPORTANT: Replace this with the path to your PDF file
    PDF_FILE_PATH = "research_paper.pdf" 
    
    if not os.path.exists(PDF_FILE_PATH):
        print(f"ERROR: PDF file not found at '{PDF_FILE_PATH}'.")
        print("Please place a PDF in the same directory and rename it to 'research_paper.pdf' (or update the path).")
        exit()

    try:
        # Step 1: Initialize the Vector Store (Load, Split, Embed via HF API)
        initialize_vector_store(PDF_FILE_PATH)
        
        # Retrieve all original chunks with metadata for citation lookup
        # This is duplicated logic to ensure we have the chunk map for the final step
        temp_loader = PyPDFLoader(PDF_FILE_PATH)
        temp_docs = temp_loader.load()
        temp_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_chunks_for_lookup = temp_splitter.split_documents(temp_docs)
        for i, doc in enumerate(all_chunks_for_lookup):
            doc.metadata["chunk_id"] = i
            
        # Step 2: Create the RAG Chain
        rag_chain = create_rag_chain()

        # Step 3: Ask a Question
        user_question = input("\nEnter your question about the PDF: ")
        
        print("\n--- Generating Response via Hugging Face API ---\n")
        
        result = query_pdf(user_question, rag_chain, all_chunks_for_lookup)

        # Step 4: Display Results
        print("💡 **Answer**:")
        print(result['answer'])
        print("\n--- Citations (Source Details) ---")
        if result['citations']:
            # Use set() to ensure unique citations are displayed
            for citation in set(result['citations']):
                print(f"• {citation}")
        else:
            print("No relevant citations found in the document for this answer.")
        print("\n--- Suggested Follow-Up Questions ---")
        if result['followups']:
          for q in result['followups']:
           print(f"👉 {q}")
        else:
           print("No follow-up suggestions available.") 
        print("\n------------------------------\n")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your Hugging Face API key and network connection.")