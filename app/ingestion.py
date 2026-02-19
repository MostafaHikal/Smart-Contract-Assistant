import os
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Universal Loader (PDF & DOCX)
def load_and_chunk_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    
    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    data = loader.load()
    
    # We kept your original chunking settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# Step 2: Embed and Store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    folder_path = "vector_store/faiss_index"
    os.makedirs("vector_store", exist_ok=True)
    vector_db.save_local(folder_path)
    
    return folder_path

# Step 3: Main function to be called by the UI
def ingest_document(file_path):
    """
    Orchestrates the loading and indexing of a new document.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File {file_path} not found."
            
        chunks = load_and_chunk_file(file_path)
        path = create_vector_store(chunks)
        return "UPLOADED"
    except Exception as e:
        return f"Ingestion Error: {str(e)}"

if __name__ == "__main__":
    # Test script
    test_file = "data/Smart_Contract_Assistant_Spec.docx.pdf" 
    result = ingest_document(test_file)
    print(result)