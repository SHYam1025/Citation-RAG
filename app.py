import os
import json
import streamlit as st
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# This line loads the variables from the .env file into os.environ
load_dotenv()

# LangChain components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
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

# --- 3. Streamlit Page Config ---
st.set_page_config(page_title="PDF RAG with Citations", page_icon="📄", layout="wide")

# Initialize session state for vector store if it doesn't exist
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --- 4. RAG Pipeline Functions ---

def initialize_vector_store(pdf_path: str):
    """Loads PDF, splits it, and creates a FAISS vector store using HF Embeddings."""
    
    # 1. Load Documents
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split and Assign ID
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    
    # Remove empty chunks silently instead of throwing a hard error
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    if not chunks:
        raise ValueError("No extractable text could be found using standard PDF extraction. The PDF might be fully scanned (image-only) or contain secured, unreadable font encodings. Since you requested NO OCR, this PDF currently cannot be read into vector embeddings.")

    # Assign a unique, simple integer ID to each chunk and store them globally
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
        
    st.session_state.all_chunks = chunks

    # 3. Create Vector Store with local Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )

    try:
        st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        raise ValueError(f"Failed to generate embeddings: {str(e)}")
        
    return len(chunks)


def format_docs_with_id(docs: List[Document]) -> str:
    """Formats retrieved documents to include a citation Source ID and content."""
    formatted = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        page_num = doc.metadata.get("page", 0)
        if isinstance(page_num, int):
            page_num += 1 # Make 1-indexed
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
    if st.session_state.vector_store is None:
        raise ValueError("Vector store not initialized. Upload a PDF first.")

    retriever = st.session_state.vector_store.as_retriever(k=5)

    llm = ChatGroq(
        model_name=LLM_REPO_ID,
        temperature=0.1,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    output_parser = JsonOutputParser(pydantic_object=CitedAnswer)
    format_instructions = output_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    system_prompt = (
        "You are a helpful and expert research assistant. "
        "Your task is to answer a user's question using only the provided documents. "
        "Do not include any information from your pre-existing knowledge base.\n\n"
        
        f"JSON OUTPUT FORMAT:\n{format_instructions}\n\n"
        
        "Your final output should be ONLY the JSON object, with no extra text or conversational filler.\n"
        "The `citations` field must be an array of integers representing the specific Source IDs used for your answer.\n\n"
        
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Response (in JSON format):"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{question}"),
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

def generate_followups(question: str, answer: str) -> List[str]:
    """Generate 3 natural follow-up questions based on the Q&A."""
    llm = ChatGroq(
        model_name=LLM_REPO_ID,
        temperature=0.5,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    followup_prompt = (
        "Suggest 3 follow-up questions based on the answer below. "
        "Format as numbered list strictly.\n"
        f"Question: {question}\nAnswer: {answer}"
    )
    response = llm.invoke(followup_prompt)
    return [q.split(". ", 1)[-1].strip() for q in response.content.split("\n") if q.strip().startswith(('1.', '2.', '3.'))]

def query_pdf(question: str) -> Dict[str, Any]:
    """Invokes the RAG chain and formats the final output, including paragraph text."""
    chain = create_rag_chain()
    
    # 1. Invoke the chain to get the structured answer
    try:
        structured_response = chain.invoke(question)
        
        # Parse the output flexibly in case parser falls back to dict
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
         return {"answer": f"Error in processing response: {e}", "citations": [], "followups": []}

    followups = generate_followups(question, final_answer)

    # 2. Look up full citation details and content from the original chunks
    all_chunks_map = {doc.metadata['chunk_id']: doc for doc in st.session_state.all_chunks}
    
    # Use a set to get unique chunk IDs before sorting
    unique_cited_ids = sorted(list(set(cited_ids)))
    
    formatted_citations = []
    for chunk_id in unique_cited_ids:
        chunk = all_chunks_map.get(chunk_id)
        if chunk:
            metadata = chunk.metadata
            source = os.path.basename(metadata.get("source", "PDF"))
            page_num = metadata.get("page", 0)
            if isinstance(page_num, int):
                page_num += 1
                
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
        "citations": formatted_citations,
        "followups": followups
    }

# --- 5. Streamlit App Layout ---

st.title("📄 Citations RAG Interface")
st.markdown("Upload a PDF document and ask questions to receive AI-powered answers backed by **exact document citations** and **suggested follow-ups**.")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing PDF and generating embeddings..."):
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_pdf_path = tmp_file.name
                
                try:
                    num_chunks = initialize_vector_store(tmp_pdf_path)
                    st.success(f"Successfully processed {num_chunks} chunks!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    # Clean up the temp file
                    os.unlink(tmp_pdf_path)
                    
    st.divider()
    st.caption("Powered by LangChain, Hugging Face Embeddings, & Groq LLMs")

# Main Chat Interface
st.header("2. Chat with your Data")

if st.session_state.vector_store is None:
    st.info("👈 Please upload and process a PDF document in the sidebar to start asking questions.")
else:
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            if chat["citations"]:
                with st.expander("📚 View Source Citations", expanded=False):
                    for idx, citation in enumerate(chat["citations"]):
                        st.markdown(f"**Source {idx + 1}: {citation['source']} (Page {citation['page']})**")
                        st.markdown("**Paragraph:**")
                        st.info(citation['content'])
                        st.caption(f"Chunk ID: {citation['chunk_id']}")
            
            if chat.get("followups"):
                st.markdown("**Suggested Follow-up Questions:**")
                for q in chat["followups"]:
                    st.markdown(f"👉 *{q}*")

    # Handle Follow-up Button Actions with Streamlit callbacks
    def set_question(q):
        st.session_state.current_question = q

    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
        
    last_chat_followups = []
    if len(st.session_state.chat_history) > 0:
        last_chat_followups = st.session_state.chat_history[-1].get("followups", [])

    if last_chat_followups:
        st.write("Click a follow-up to ask:")
        for idx, followup in enumerate(last_chat_followups):
            st.button(followup, key=f"btn_{idx}", on_click=set_question, args=(followup,))

    # Input for new question
    question_input = st.chat_input("Ask a question about your document...")
    
    # Decide which query to process
    query_to_process = None
    if st.session_state.current_question:
        query_to_process = st.session_state.current_question
        st.session_state.current_question = "" # Reset
    elif question_input:
        query_to_process = question_input

    if query_to_process:
        with st.chat_message("user"):
            st.markdown(query_to_process)
            
        with st.chat_message("assistant"):
            with st.spinner("Searching document and thinking..."):
                response = query_pdf(query_to_process)
                
            st.markdown(response["answer"])
            
            if response["citations"]:
                with st.expander("📚 View Source Citations", expanded=True):
                    for idx, citation in enumerate(response["citations"]):
                        st.markdown(f"**Source {idx + 1}: {citation['source']} (Page {citation['page']})**")
                        st.markdown("**Paragraph:**")
                        st.info(citation['content'])
                        st.caption(f"Chunk ID: {citation['chunk_id']}")
            else:
                st.info("No relevant citations found in the document for this answer.")
                
            if response.get("followups"):
                st.markdown("**Suggested Follow-up Questions:**")
                for q in response["followups"]:
                    st.markdown(f"👉 *{q}*")
            
            # Save to chat history
            st.session_state.chat_history.append({
                "question": query_to_process,
                "answer": response["answer"],
                "citations": response["citations"],
                "followups": response.get("followups", [])
            })
            st.rerun()
