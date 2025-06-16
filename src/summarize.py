import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def make_summarizer(vector_store, model: str, temp: float, max_tokens: int):
    llm = ChatGroq(
        model=model,
        temperature=temp,
        max_tokens=max_tokens,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    return chain
