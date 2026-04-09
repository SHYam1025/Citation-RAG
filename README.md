# ⚡ Citation RAG — High-Precision Document Intelligence System

A production-grade Retrieval-Augmented Generation (RAG) system for querying long PDF documents with **exact page & paragraph citations** and **AI-generated follow-up questions**.

> Chat with your documents like a research assistant — with verifiable answers.

---

## 🚀 Key Features

- 💬 **Context-Aware Q&A**
  - Ask questions in natural language
  - Get precise answers grounded in document context

- 📌 **Exact Citations**
  - Returns **page number + paragraph-level references**
  - Eliminates hallucination risk

- 🔁 **Follow-Up Question Generation**
  - AI suggests relevant next questions to deepen exploration

- 📄 **Multi-PDF Support**
  - Upload and query multiple documents seamlessly

- ⚡ **Streaming Responses**
  - Real-time answer generation for better UX

---

## 🧠 System Architecture
User Query
↓
Embedding Model
↓
Vector Database (FAISS / Chroma)
↓
Relevant Chunk Retrieval
↓
LLM (Gemini / OpenAI)
↓
Answer + Citations + Follow-ups

<img width="1385" height="881" alt="Demo-Screenshot" src="https://github.com/user-attachments/assets/9ea3b8bd-f244-4f3d-967c-859a0e0e2aac" />
<img width="1390" height="896" alt="Screenshot 2026-02-21 at 5 45 26 PM" src="https://github.com/user-attachments/assets/7f3ec1cf-824d-4417-b78f-a1fece2dd6ec" />

---

## 🛠️ Tech Stack

| Layer            | Technology Used                          |
|------------------|------------------------------------------|
| Backend          | FastAPI (Python)                         |
| LLM              | Google Gemini / OpenAI                   |
| Embeddings       | Sentence Transformers / OpenAI Embeddings|
| Vector Store     | FAISS / ChromaDB                         |
| Document Parsing | PyPDF / LangChain                        |
| Streaming        | FastAPI Streaming / Async APIs           |
| Frontend         | HTML, CSS, JavaScript                    |

---

## 📂 Project Structure

```bash
Citation-RAG/
├── app.py                     # Main application
├── fast-main.py              # FastAPI server
├── RAG_with_citation.py      # Core RAG pipeline
├── streaming_RAG.py          # Streaming responses
├── uploaded_pdfs/            # Document storage
├── research_paper.pdf        # Sample file
├── requirements.txt
└── README.md

```
---

## ⚙️ How It Works

1. 📥 Upload PDF documents  
2. ✂️ Split into semantic chunks  
3. 🔢 Convert chunks into embeddings  
4. 🧠 Store in vector database  
5. 🔍 Retrieve relevant chunks for query  
6. 🤖 LLM generates answer with citations  
7. 💡 Suggests intelligent follow-up questions  

---

## 🧪 Example Output

**Query:**  
> How does self-attention work in transformers?

**Response:**  
- Answer generated using retrieved context  
- 📌 Source: Page 2, Paragraph 3  
- 📌 Source: Page 6, Paragraph 1  

**Follow-Up Suggestions:**
- What are attention heads?  
- How does scaling affect performance?  

---

## 📦 Installation

```bash
git clone https://github.com/your-username/Citation-RAG.git
cd Citation-RAG
```
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```
pip install -r requirements.txt
```
🔑 Environment Setup

Create a .env file:
```env
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key   # optional
```
## ▶️ Run the Application

```bash
python fast-main.py
```
Open in browser:
```
http://localhost:8000
```
📊 Use Cases
📚 Research paper analysis
⚖️ Legal document querying
📑 Technical documentation assistant
🧾 Policy & compliance search
📊 Enterprise knowledge base
🔥 Highlights
Reduces hallucination using citation-backed answers
Improves research efficiency with AI follow-ups
Handles long-context documents effectively
Built with scalable RAG architecture
📈 Future Improvements
Multi-document ranking optimization
Hybrid search (BM25 + vector search)
UI improvements (chat memory, history)
Deployment on cloud (AWS / GCP)
📄 License

MIT License

👨‍💻 Author

Shyam Pathak
AI Engineer | Generative AI | LLM Systems

⭐ If you found this useful, consider giving it a star!


---

# 💡 What this fixes
- ✅ Proper code block closing  
- ✅ Clean section hierarchy  
- ✅ Bullet formatting (important for readability)  
- ✅ GitHub renders it cleanly  

---

# 🎯 Straight truth
Right now you're very close —  
but small formatting mistakes make your repo look **amateur**.

This fixed version:
👉 Looks like a **serious open-source project**

---

If you want to go one level higher:
I can add:
- 🔥 Demo GIF (this alone boosts impressions a lot)
- 🔥 “Architecture diagram” section
- 🔥 Recruiter-focused highlights section

Built by: **"SHYAM PATHAK(SHYam1025)”** 🚀
