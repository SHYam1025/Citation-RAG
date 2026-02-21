import sys
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader
import fitz

pdf_path = "EOT-Al_Namaa_Poultry-May_2023.pdf"

try:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    print("PyMuPDF fitz raw extract length:", len(text))
    
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print("PyMuPDFLoader docs length:", len(docs))
    print("Content length:", sum(len(d.page_content) for d in docs))
except Exception as e:
    print("Error:", e)
