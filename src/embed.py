from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def build_or_load_index(chunks: list[str]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(docs, embedding=embeddings)
