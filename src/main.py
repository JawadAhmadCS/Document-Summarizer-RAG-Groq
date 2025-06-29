import sys
import os
from PIL import Image
import numpy as np
import tempfile
import fitz  # PyMuPDF
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from embed import build_or_load_index
from summarize import make_summarizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ocr_utils import extract as extract_text_from_image

# 🎨 Page config
st.set_page_config(page_title="Document Summarizer", page_icon="🧠", layout="centered")

# 🎨 Background
page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #f0f0f0, #d6e4ff);
}
[data-testid="stHeader"] {
    background-color: rgba(255,255,255,0);
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# 🧠 Headings
st.markdown("<h1 style='text-align: center;'>📄 Document Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a document and get a concise AI-powered summary using RAG + Groq</p>", unsafe_allow_html=True)


# 🎞️ Lottie Animation
def load_lottie_animation(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_j1adxtyb.json"
lottie_animation = load_lottie_animation(lottie_url)
if lottie_animation:
    st_lottie(lottie_animation, height=180, key="header_lottie")


# 📥 File uploader
#uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md", "png", "jpg", "jpeg"])

def reuad_uploaded_file(uploaded_file):
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
def read_uploaded_file(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    
    if ext in ["pdf"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        doc = fitz.open(tmp_file_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
    
    elif ext in ["png", "jpg", "jpeg"]:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        text = extract_text_from_image(image_np)

    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    
    return text


def chunk_texts(text):
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
    )
    result = summarizer.invoke({"query": prompt})
    return result["result"]

# 📄 If document uploaded
if uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()
    
    if ext in ["png", "jpg", "jpeg"]:
        st.warning("⚠️ Note: This model is **not fully trained** for extracting text from images. It may produce inaccurate results. For best performance, use PDF, TXT, or MD files.")


    text = read_uploaded_file(uploaded_file)

    st.markdown("### 📜 Extracted Text")
    with st.expander("Click to preview extracted content"):
        st.text_area("Document Text", text[:4000] + "..." if len(text) > 4000 else text, height=300)

    prompt = st.text_input("🔍 Prompt", value="Summarize the document")

    if st.button("🚀 Generate Summary"):
        with st.spinner("⏳ Summarizing... Please wait"):
            try:
                summary = generate_summary(
                    text=text,
                    prompt=prompt,
                    model=st.secrets["GROQ_CHAT_MODEL"],
                    temp=float(st.secrets["LLM_TEMP"]),
                    max_tokens=int(st.secrets["LLM_MAX_TOKENS"]),
                    api_key=st.secrets["GROQ_API_KEY"]
                )
                st.subheader("🧠 Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("👆 Upload a document above to get started.")


# there can be spelling mistake in this text words think what it can be