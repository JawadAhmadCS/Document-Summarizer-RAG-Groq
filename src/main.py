import streamlit as st
import tempfile
import fitz  # PyMuPDF
from embed import build_or_load_index
from summarize import make_summarizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def chunk_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(texts[0])  # Just one file for now

def main():
    st.set_page_config(page_title="Document Summarizer RAG")
    st.title("📄 Document Summarizer using RAG")

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])

    if uploaded_file is not None:
        file_text = read_uploaded_file(uploaded_file)

        st.markdown("### 📜 Extracted Text")
        with st.expander("Click to preview extracted content"):
            st.write(file_text[:1500] + "..." if len(file_text) > 1500 else file_text)

        prompt = st.text_input("Prompt:", value="Summarize the document")

        if st.button("🚀 Generate Summary"):
            with st.spinner("Embedding and summarizing..."):
                chunks = chunk_texts([file_text])
                vector_store = build_or_load_index(chunks)

                summarizer = make_summarizer(
                    vector_store,
                    model=st.secrets["GROQ_CHAT_MODEL"],
                    temp=0.2,
                    max_tokens=512
                )

                result = summarizer.invoke({"query": prompt})
                st.subheader("🧠 Summary")
                st.write(result['result'])

if __name__ == "__main__":
    main()
