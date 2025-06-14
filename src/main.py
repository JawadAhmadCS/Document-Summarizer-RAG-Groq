import streamlit as st
from embed import embed_uploaded_text
from summarize import make_summarizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile

def chunk_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_text(texts[0])  # Just one file for now

def main():
    st.set_page_config(page_title="Document Summarizer RAG")
    st.title("📄 Document Summarizer using RAG")

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])

    if uploaded_file is not None:
        file_text = uploaded_file.read().decode("utf-8", errors="ignore")

        st.subheader("📄 Extracted Text")
        st.text_area("File Content", file_text[:2000], height=200)

        chunks = chunk_texts([file_text])
        vector_store = embed_uploaded_text(chunks)

        summarizer = make_summarizer(vector_store)

        if st.button("Summarize the document"):
            with st.spinner("Running summarizer..."):
                question = "Summarize the document"
                result = summarizer.invoke({"query": question})
                st.subheader("🧠 Summary")
                st.write(result['result'])

if __name__ == "__main__":
    main()
