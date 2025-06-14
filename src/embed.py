from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def embed_uploaded_text(chunks: list[str]):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content=chunk) for chunk in chunks]
    vs = FAISS.from_documents(docs, embedding=emb)
    return vs
