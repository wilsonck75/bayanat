import os
import glob
from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from enferno.tasks import celery
from enferno.rag.utils import ImageLoader
from flask import current_app

@celery.task
def ingest_documents(data_dir=None):
    """
    Celery task to ingest documents from a directory into ChromaDB.
    """
    if data_dir is None:
        # Default to a configured data directory
        data_dir = current_app.config.get("RAG_DATA_DIR", "backend/data")
    
    db_dir = current_app.config.get("CHROMA_DB_DIR", "backen/chroma_db")
    embedding_model = current_app.config.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    
    print(f"Starting ingestion from {data_dir} to {db_dir}...")
    
    documents = []
    
    # Load .txt files
    for txt_path in glob.glob(os.path.join(data_dir, "*.txt")):
        try:
            loader = TextLoader(txt_path, encoding="utf-8")
            documents.extend(loader.load())
            print(f"Loaded {txt_path}")
        except Exception as e:
            print(f"Error loading {txt_path}: {e}")

    # Load .pdf files
    for pdf_path in glob.glob(os.path.join(data_dir, "*.pdf")):
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            print(f"Loaded {pdf_path}")
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")

    # Load images
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    for ext in image_extensions:
        for img_path in glob.glob(os.path.join(data_dir, ext)):
            try:
                print(f"Processing image {img_path} with Vision Model...")
                loader = ImageLoader(img_path)
                documents.extend(loader.load())
                print(f"Loaded transcription from {img_path}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
            
    if not documents:
        print("No documents found.")
        return "No documents found."

    print(f"Splitting {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    print("Creating vector store...")
    vector_store = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name="arabic_rag"
    )
    
    vector_store.add_documents(documents=chunks)
    print(f"Ingestion complete. Vector store saved to {db_dir}")
    return f"Ingestion complete. Processed {len(chunks)} chunks."
