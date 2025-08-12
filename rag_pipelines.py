from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA


def get_local_embeddings():
    """Initializes and returns the embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_gemini_embeddings():
    """Initializes and returns the Google Generative AI embedding model."""
    print("Loading Google embedding model: models/embedding-001")
    # Note: This requires GOOGLE_API_KEY to be set.
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def create_standard_rag_chain(documents, llm, embeddings):
    """Pipeline 1: Standard recursive chunking."""
    print("Building Standard RAG chain...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def create_parent_document_rag_chain(documents, llm, embeddings):
    """Pipeline 2: Late Chunking (Parent Document Retriever)."""
    print("Building Parent Document RAG chain...")
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    vectorstore = FAISS.from_documents(documents, embeddings)
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    retriever.add_documents(documents)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def create_sentence_window_rag_chain(documents, llm, embeddings):
    """Pipeline 3: Sentence Window Retriever."""
    print("Building Sentence Window RAG chain...")

    # Use a token-aware sentence splitter
    text_splitter = SentenceTransformersTokenTextSplitter(
        model_name="all-MiniLM-L6-v2",
        tokens_per_chunk=256,  # Max tokens per chunk
        chunk_overlap=50,
    )

    docs = text_splitter.split_documents(documents)

    # Add doc_id to each document
    for i, doc in enumerate(docs):
        doc.metadata["doc_id"] = i

    vectorstore = FAISS.from_documents(docs, embeddings)

    # Retrieve more surrounding sentences to simulate the "window"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
