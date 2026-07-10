import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(uploaded_file):
    temp_path = r"C:\Users\vasil\Downloads\cloud.txt"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader(temp_path)
    pages = loader.load_and_split()
    return pages, temp_path