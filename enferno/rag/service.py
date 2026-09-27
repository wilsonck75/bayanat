from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from flask import current_app
from enferno.settings import Config

class ArabicRAG:
    """
    Service class for Arabic RAG operations.
    Adapter for LangChain RAG pipeline.
    """
    def __init__(self, llm=None):
        self.embedding_model_name = current_app.config.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
        self.db_dir = current_app.config.get("CHROMA_DB_DIR", "chroma_db")
        
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=self.db_dir,
            embedding_function=self.embeddings,
            collection_name="arabic_rag"
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Use GPT-4o for better multilingual performance
        # Ensure OPENAI_API_KEY is in environment
        self.llm = llm if llm else ChatOpenAI(model="gpt-4o", temperature=0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent knowledge assistant. Your goal is to answer questions accurately based ONLY on the provided context.
            
            STRICT OUTPUT FORMAT:
            You must provide every answer in TWO languages:
            1. English Answer
            2. Arabic Answer (Translation)
            
            Use the following format exactly:
            **English Answer:**
            [Your answer here]
            
            **الإجابة بالعربية:**
            [Translation here]
            
            If the user greets you (e.g., "Hello"), greet them back in both languages politely.
            If the context does not contain the answer, state that clearly in both languages.
            Do not echo the user's question."""),
            ("human", """Context:
            {context}
            
            Question:
            {question}""")
        ])

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_chain(self):
        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def query(self, question: str):
        chain = self.get_chain()
        return chain.invoke(question)
