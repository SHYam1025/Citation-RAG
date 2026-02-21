# main.py
import os
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

# LangChain components
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
#from langchain_community.chains import RetrievalQA
# Load environment variables
load_dotenv()

app = FastAPI(title="RAG PDF QA System")

# --- Pydantic Models ---
class QueryModel(BaseModel):
    question: str = Field(..., description="The question to ask the RAG system")

class CitedAnswer(BaseModel):
    answer: str
    citations: List[int]

# --- Global Variables ---
VECTOR_STORE = None
ALL_CHUNKS_FOR_LOOKUP: List[Document] = []
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPLITTER_CONFIG = {
    "chunk_size": 900,
    "chunk_overlap": 120,
    "separators": ["\n\n", "\n", " ", ""],
    "add_start_index": True
}

# --- Helper Functions ---
def initialize_vector_store(pdf_path: str):
    """Load PDF, split, embed, and create FAISS vector store"""
    global VECTOR_STORE, ALL_CHUNKS_FOR_LOOKUP
    
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(**SPLITTER_CONFIG)
    chunks = text_splitter.split_documents(documents)
    
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
        
    ALL_CHUNKS_FOR_LOOKUP = chunks
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    
    VECTOR_STORE = FAISS.from_documents(chunks, embeddings)
    return len(chunks)

def format_docs_with_id(docs: List[Document]) -> str:
    """Format chunks for LLM prompt"""
    formatted = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        page_index = doc.metadata.get("page", -1)
        page_num = int(page_index) + 1 if isinstance(page_index, int) and page_index >= 0 else "N/A"
        source = os.path.basename(doc.metadata.get("source", "PDF_Source"))
        formatted.append(
            f"Source ID: {chunk_id}\nSource: {source} (Page {page_num})\nContent: {doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)

def create_rag_chain():
    """Create the LangChain RAG chain"""
    if VECTOR_STORE is None:
        raise ValueError("Vector store not initialized.")
        
    retriever = VECTOR_STORE.as_retriever(k=10)
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    output_parser = JsonOutputParser(pydantic_object=CitedAnswer)
    escaped_format_instructions = output_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    
    system_prompt_text = (
        "You are a helpful research assistant. Use only provided documents.\n\n"
        f"JSON OUTPUT FORMAT:\n{escaped_format_instructions}\n\n"
        "Your final output should be ONLY the JSON object, with no extra text or conversational filler.\n"
        "The `citations` field must be an array of integers representing the specific Source IDs used for your answer.\n\n"
        "Context:\n{context}\n\n"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            ("user", "Question: {question}")
        ]
    )
    
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs_with_id),
         "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | output_parser
    )
    
    return rag_chain

def generate_followups(question: str, answer: str, llm) -> List[str]:
    """Generate 3 follow-up questions"""
    prompt = (
        "Suggest 3 follow-up questions based on the answer below. "
        "Format as numbered list.\n"
        f"Question: {question}\nAnswer: {answer}"
    )
    response = llm.invoke(prompt)
    return [q.split(". ", 1)[-1].strip() for q in response.content.split("\n") if q.strip().startswith(('1.', '2.', '3.'))]

def query_pdf(question: str, rag_chain, all_chunks: List[Document]) -> Dict[str, Any]:
    """Run RAG chain and return structured JSON"""
    try:
        structured_response = rag_chain.invoke(question)
        if isinstance(structured_response, dict):
            final_answer = structured_response.get("answer", str(structured_response))
            citations_raw = structured_response.get("citations", [])
            cited_ids = []
            for c in citations_raw:
                if isinstance(c, dict):
                    cited_ids.append(c.get("chunk_id", c.get("source_id")))
                elif isinstance(c, int):
                    cited_ids.append(c)
                elif isinstance(c, str) and c.isdigit():
                    cited_ids.append(int(c))
            cited_ids = [cid for cid in cited_ids if cid is not None]
        elif hasattr(structured_response, 'answer'):
            final_answer = structured_response.answer
            cited_ids = getattr(structured_response, 'citations', [])
        else:
            return {"answer": str(structured_response), "citations": [], "followups": []}
    except Exception as e:
        return {"answer": f"Error: {type(e).__name__}: {e}", "citations": [], "followups": []}
    
    followups = generate_followups(question, final_answer, rag_chain.steps[-2])
    
    all_chunks_map = {doc.metadata['chunk_id']: doc.metadata for doc in all_chunks}
    formatted_citations = []
    for chunk_id in sorted(set(cited_ids)):
        metadata = all_chunks_map.get(chunk_id)
        if metadata:
            source = os.path.basename(metadata.get("source", "PDF"))
            page_index = metadata.get("page", -1)
            page_num = int(page_index) + 1 if isinstance(page_index, int) and page_index >= 0 else "N/A"
            formatted_citations.append(f"{source} (Page {page_num}, Chunk ID: {chunk_id})")
    
    return {"answer": final_answer, "citations": formatted_citations, "followups": followups}

# --- FastAPI Routes ---
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF"""
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}
    
    save_path = Path("uploaded_pdfs") / file.filename
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    
    chunks_count = initialize_vector_store(str(save_path))
    return {"message": f"PDF uploaded and processed successfully ({chunks_count} chunks)."}

@app.post("/ask")
def ask_question(query: QueryModel):
    """Ask a question about the uploaded PDF"""
    if VECTOR_STORE is None:
        return {"error": "No PDF uploaded yet."}
    
    rag_chain = create_rag_chain()
    return query_pdf(query.question, rag_chain, ALL_CHUNKS_FOR_LOOKUP)
