import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# This line loads the variables from the .env file into os.environ
load_dotenv()

from pydantic import BaseModel, Field

# LangChain components
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser

# --- 1. Define Citation Schema (Pydantic) ---

class CitedAnswer(BaseModel):
    """The final answer and the specific sources used to generate it."""
    answer: str = Field(..., description="The complete, direct answer to the user question, based ONLY on the provided context.")
    citations: List[int] = Field(..., description="The integer IDs of the SPECIFIC source chunks which justify the answer.")

# --- 2. Configuration ---

LLM_REPO_ID = "llama-3.3-70b-versatile"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Global variables
VECTOR_STORE = None
ALL_CHUNKS_FOR_LOOKUP = []

# --- 3. RAG Pipeline Functions ---

def initialize_vector_store(pdf_path: str):
    """Loads PDF, splits it, and creates a FAISS vector store using HF Embeddings."""
    global VECTOR_STORE, ALL_CHUNKS_FOR_LOOKUP

    # 1. Load Documents
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split and Assign ID
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)

    # Assign a unique, simple integer ID to each chunk and store them globally
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
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
    """Constructs the LangChain RAG chain using the Groq LLM and JSON parser."""
    if VECTOR_STORE is None:
        raise ValueError("Vector store not initialized. Run initialize_vector_store first.")

    retriever = VECTOR_STORE.as_retriever(k=5)

    llm = ChatGroq(
        model_name=LLM_REPO_ID,
        temperature=0.1,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    output_parser = JsonOutputParser(pydantic_object=CitedAnswer)
    format_instructions = output_parser.get_format_instructions()

    system_prompt = (
        "You are a helpful and expert research assistant. "
        "Your task is to answer a user's question using only the provided documents. "
        "Do not include any information from your pre-existing knowledge base.\n\n"
        
        "Your response must be a single, structured JSON object with two keys: `answer` and `citations`.\n\n"
        
        "The `answer` field must contain the full answer to the question.\n"
        "The `citations` field must be an array of integers, where each integer corresponds to the `Source ID` of the chunk which supports a statement in your answer.\n"
        "If the answer is not available in the provided documents, the `answer` field should be 'Answer not available in the provided document,' and the `citations` array should be empty.\n\n"
        
        "Your final output should be ONLY the JSON object, with no extra text or conversational filler.\n\n"
        "Formatting instructions:\n{format_instructions}\n"
        
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Response (in JSON format):"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{question}"),
        ]
    ).partial(format_instructions=format_instructions)

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs_with_id),
         "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | output_parser
    )
    return rag_chain


def query_pdf(question: str, rag_chain: Any) -> Dict[str, Any]:
    """Invokes the RAG chain and formats the final output, including paragraph text."""
    
    # 1. Invoke the chain to get the structured answer
    try:
        structured_response = rag_chain.invoke(question)
        
        # Check if the response is a dictionary (in case of parsing failure)
        if isinstance(structured_response, dict):
            # Fallback to dictionary key access
            final_answer = structured_response.get("answer", "Answer not available in the provided document.")
            cited_ids = structured_response.get("citations", [])
            
        elif hasattr(structured_response, 'answer') and hasattr(structured_response, 'citations'):
            # The ideal case: an object with the expected attributes is returned
            final_answer = structured_response.answer
            cited_ids = structured_response.citations
        else:
            # Handle any other unexpected output types
            return {"answer": "An unexpected response format was received.", "citations": []}
            
    except Exception as e:
        # Handle cases where the LLM returns completely unparseable text
        return {"answer": f"Error in structured response: {e}", "citations": []}

    # 2. Look up full citation details and content from the original chunks
    all_chunks_map = {doc.metadata['chunk_id']: doc for doc in ALL_CHUNKS_FOR_LOOKUP}
    
    # Use a set to get unique chunk IDs before sorting
    unique_cited_ids = sorted(list(set(cited_ids)))
    
    formatted_citations = []
    for chunk_id in unique_cited_ids:
        chunk = all_chunks_map.get(chunk_id)
        if chunk:
            metadata = chunk.metadata
            source = os.path.basename(metadata.get("source", "PDF"))
            page_num = metadata.get("page", "N/A")
            # Get the content of the chunk
            content = chunk.page_content.strip()
            
            formatted_citations.append({
                "source": source,
                "page": page_num,
                "chunk_id": chunk_id,
                "content": content
            })

    # 3. Return the formatted result
    return {
        "answer": final_answer,
        "citations": formatted_citations
    }

# --- 4. Main Execution ---

if __name__ == "__main__":
    
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: Please set the GROQ_API_KEY environment variable.")
        exit()
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        print("ERROR: Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
        exit()

    PDF_FILE_PATH = "research_paper.pdf" 
    
    if not os.path.exists(PDF_FILE_PATH):
        print(f"ERROR: PDF file not found at '{PDF_FILE_PATH}'.")
        print("Please place a PDF in the same directory and rename it to 'research_paper.pdf' (or update the path).")
        exit()

    try:
        # Step 1: Initialize the Vector Store
        initialize_vector_store(PDF_FILE_PATH)
        
        # Step 2: Create the RAG Chain
        rag_chain = create_rag_chain()

        # Step 3: Ask a Question
        user_question = input("\nEnter your question about the PDF: ")
        
        print("\n--- Generating Response via LLM API ---\n")
        
        result = query_pdf(user_question, rag_chain)

        # Step 4: Display Results (corrected to always show the answer)
        print("💡 **Answer**:")
        print(result['answer'])
        if result['citations']:
            print("\n--- Citations (Source Details) ---")
            for citation in result['citations']:
                print(f"• Source: {citation['source']} (Page {citation['page']}, Chunk ID: {citation['chunk_id']})")
                print("  --- Paragraph ---")
                print(citation['content'])
                print("------------------------------------------------------------------------------")
        else:
            print("\nNo relevant citations found in the document for this answer.")
        
        print("\n------------------------------------------------------------------------------\n")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your API keys and network connection.")