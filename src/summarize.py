from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

def make_summarizer(vector_store, model: str, temp: float, max_tokens: int, api_key: str):
    llm = ChatGroq(
        model=model,
        temperature=temp,
        max_tokens=max_tokens,
        api_key=api_key
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    return chain
