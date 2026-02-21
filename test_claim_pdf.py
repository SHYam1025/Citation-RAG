import sys
from langchain_community.document_loaders import PyMuPDFLoader
import pdfplumber

pdf_path = "uploaded_pdfs/1934-_EOT_12_-Claim_for_additional_time_due_to_stoppage_of_Works_(Interim_EOT-12).pdf"

try:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print("PyMuPDF length:", sum(len(d.page_content.strip()) for d in docs))
except Exception as e:
    print("PyMuPDF error", e)

try:
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            t = page.extract_text()
            if t: text += t
        print("pdfplumber length:", len(text))
except Exception as e:
    print("pdfplumber error", e)
