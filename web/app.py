import streamlit as st
from src.ingest import chunk_texts
from src.embed import build_or_load_index
from src.summarize import make_summarizer
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF

load_dotenv()

st.set_page_config(page_title="Document Summarizer", page_icon="🧠")

st.title("📄 Document Summarizer using RAG + Groq")

# Upload document
uploaded_file = st.file_uploader("Upload a document (.pdf, .txt, .md)", type=["pdf", "txt", "md"])

if uploaded_file:
    # Read file content
    ext = uploaded_file.name.split('.')[-1].lower()

    if ext == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    # Show preview
    with st.expander("📜 Document Preview"):
        st.write(text[:1500] + "..." if len(text) > 1500 else text)

    # Prompt input
    default_prompt = "Summarize the document"
    prompt = st.text_input("Prompt:", value=default_prompt)

    if st.button("🚀 Generate Summary"):
        with st.spinner("Embedding and summarizing..."):
            # Step 1: Chunk text
            chunks = chunk_texts([text])

            # Step 2: Build vector index (in-memory only)
            vs = build_or_load_index(chunks, vector_path="outputs/faiss_index")

            # Step 3: Create summarizer
            summarizer = make_summarizer(
                vs,
                model=os.getenv('GROQ_CHAT_MODEL', 'llama-3.3-70b-versatile'),
                temp=float(os.getenv('LLM_TEMP', 0.0)),
                max_tokens=int(os.getenv('LLM_MAX_TOKENS', 300))
            )

            # Step 4: Run prompt
            summary = summarizer.run(prompt)

            st.subheader("📋 Summary")
            st.write(summary)
