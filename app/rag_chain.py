from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import os
import tempfile
import re
import streamlit as st

# Model IDs from HuggingFace Hub
MODEL_DIR = "google/flan-t5-base"
EMBEDDING_DIR = "sentence-transformers/all-MiniLM-L6-v2"

# Clean repetitive answers
import re

def clean_answer(text):
    if not text.strip():
        return "⚠️ No answer could be extracted from the document."

    # Remove LaTeX math expressions
    text = re.sub(r'\$.*?\$', '', text)

    # Remove repeated sentences
    sentences = text.strip().split('. ')
    seen = set()
    unique_sentences = [s.strip() for s in sentences if s and s not in seen and not seen.add(s)]

    # Join cleaned sentences
    cleaned_text = '. '.join(unique_sentences).strip()

    # Capitalize first letter
    if cleaned_text and cleaned_text[0].islower():
        cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]

    # Add period at the end if missing
    if not cleaned_text.endswith('.'):
        cleaned_text += '.'

    return f"Answer: {cleaned_text}"

@st.cache_resource(show_spinner="🔄 Loading models and building vector store...")
def load_models_and_vectorstore(_pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(_pages)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_DIR, model_kwargs={"device": "cpu"})

    chroma_dir = os.path.join(tempfile.gettempdir(), "chroma_db")

    # Clear old Chroma cache
    if os.path.exists(chroma_dir):
       import shutil
       shutil.rmtree(chroma_dir)

    vector_db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=chroma_dir)
    # Configure retriever to return more relevant documents
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    # Custom inference function for flan-t5 model
    def generate_answer(prompt):
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Create a wrapper that mimics HuggingFacePipeline interface
    class Seq2SeqLLM:
        def __call__(self, text):
            return generate_answer(text)
    
    llm = Seq2SeqLLM()

    return retriever, llm


def qa_from_pdf(pages, query):
    retriever, llm = load_models_and_vectorstore(pages)
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    if not docs:
        return "⚠️ No relevant information found in the document."
    
    # Combine context from retrieved documents
    context = "\n".join([doc.page_content for doc in docs[:5]])
    
    # Create a clearer prompt that forces the model to use context
    prompt = f"""Answer the following question using ONLY the provided context. If the answer is not in the context, say 'Not found in document'.

Context:
{context}

Question: {query}

Answer based on the context above:"""
    
    # Generate answer
    raw_answer = llm(prompt)
    
    return clean_answer(raw_answer)