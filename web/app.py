import sys
import os
import streamlit as st
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="Document Summarizer", page_icon="🧠", layout="centered")

# Add root dir to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import read_uploaded_file, generate_summary

# 🎨 background style
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

# 📄 Title + Description
st.markdown("<h1 style='text-align: center;'>📄 Document Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a document and get a concise AI-powered summary using RAG + Groq</p>", unsafe_allow_html=True)

# 🔄 Lottie Animation
def load_lottie_animation(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_j1adxtyb.json"
lottie_animation = load_lottie_animation(lottie_url)
if lottie_animation:
    st_lottie(lottie_animation, height=180, key="header_lottie")

# 📂 Upload Document
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])

if uploaded_file:
    text = read_uploaded_file(uploaded_file)

    st.markdown("### 📜 Extracted Text")
    with st.expander("Click to preview extracted content"):
        st.text_area("Document Text", text[:4000] + "..." if len(text) > 4000 else text, height=300)

    # Prompt for summarization
    prompt = st.text_input("🔍 Prompt", value="Summarize the document")

    # 🚀 Summarize Button
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

                #st.success("✅ Summary generated successfully!")
                st.subheader("🧠 Summary")
                st.write(summary)

            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("👆 Upload a document above to get started.")
