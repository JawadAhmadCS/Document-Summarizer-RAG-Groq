from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

def make_summarizer(vector_store):
    llm = ChatGroq(
        model=os.getenv("GROQ_CHAT_MODEL", "llama3-70b-8192"), 
        api_key=os.getenv("GROQ_API_KEY")  # fetch actual key
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=False
    )
    return chain
