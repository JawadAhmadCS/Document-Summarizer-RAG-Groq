import streamlit as st
from src.ingest import chunk_texts
from src.embed import build_or_load_index
from src.summarize import make_summarizer
import fitz  # PyMuPDF

st.set_page_config(page_title="Document Summarizer", page_icon="🧠")
st.title("📄 Document Summarizer using RAG + Groq")

uploaded_file = st.file_uploader("Upload a document (.pdf, .txt, .md)", type=["pdf", "txt", "md"])

if uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()

    if ext == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    with st.expander("📜 Document Preview"):
        st.write(text[:1500] + "..." if len(text) > 1500 else text)

    prompt = st.text_input("Prompt:", value="Summarize the document")

    if st.button("🚀 Generate Summary"):
        with st.spinner("Embedding and summarizing..."):
            chunks = chunk_texts([text])

            vs = build_or_load_index(chunks)

            summarizer = make_summarizer(
                vs,
                model=st.secrets["GROQ_CHAT_MODEL"],
                temp=float(st.secrets["LLM_TEMP"]),
                max_tokens=int(st.secrets["LLM_MAX_TOKENS"])
            )

            summary = summarizer.run(prompt)

            st.subheader("📋 Summary")
            st.write(summary)
