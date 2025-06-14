### 📄 Document Summarizer using RAG + Groq

This project is a **Streamlit-based web app** that summarizes documents (`.pdf`, `.txt`, `.md`) using **Retrieval-Augmented Generation (RAG)** powered by **Groq's LLaMA 3 model**. It chunks and embeds text using **HuggingFace sentence transformers** and performs intelligent summarization via a **vector store (FAISS)** and **LLM-based retrieval pipeline**. Fast and scalable, ideal for summarizing large documents with high accuracy.

# Run in VS Code

```bash
git clone https://github.com/yourusername/document-summarizer-rag.git
cd document-summarizer-rag
pip install -r requirements.txt
```

Create a .env file and add your Groq API key:

```env
GROQ_API_KEY=your_actual_key_here
GROQ_CHAT_MODEL=llama-3.3-70b-versatile
LLM_TEMP=0.0
LLM_MAX_TOKENS=300
```

Than run this in terminal:

```
streamlit run src/main.py
```
