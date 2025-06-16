import tempfile
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embed import build_or_load_index
from src.summarize import make_summarizer
import os
import subprocess

def launch_streamlit_app():
    # Path to web/app.py
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web", "app.py"))

    # Launch Streamlit
    subprocess.run(["streamlit", "run", app_path])

def read_uploaded_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()

    if ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        doc = fitz.open(tmp_file_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    return text

def chunk_texts(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(text)

def generate_summary(text: str, prompt: str, model: str, temp: float, max_tokens: int, api_key: str):
    chunks = chunk_texts(text)
    vector_store = build_or_load_index(chunks)
    summarizer = make_summarizer(
        vector_store,
        model=model,
        temp=temp,
        max_tokens=max_tokens,
        api_key=api_key
    )
    result = summarizer.invoke({"query": prompt})
    return result["result"]

if __name__ == "__main__":
    launch_streamlit_app()
