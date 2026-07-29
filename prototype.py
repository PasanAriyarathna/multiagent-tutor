import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def build_rag_pipeline(pdf_directory="./lessons"):
    if not os.path.exists(pdf_directory):
        print("Please add at least 20 lesson PDFs to the ./lessons/ directory.")
        return None

    loader = PyPDFDirectoryLoader(pdf_directory)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

def evaluate_rag(retriever):
    test_queries = [
        "What is the definition of a variable in Python?",
        "How does a for loop work under the hood?",
        "Why is my list slicing [::-1] raising an IndexError?",
        "Explain data structures.",
        "How to train a neural network from scratch?"
    ]

    print("--- RAG Retrieval Evaluation ---")
    for q in test_queries:
        docs = retriever.invoke(q)
        print(f"\nQuery: {q}")
        for i, doc in enumerate(docs):
            print(f"Chunk {i+1}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    retriever = build_rag_pipeline()
    if retriever:
        evaluate_rag(retriever)