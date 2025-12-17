from src.document_loader import load_documents
from src.vector_store import create_vector_store
from src.rag_pipeline import build_rag_chain

def main():
    print("Loading documents...")
    documents = load_documents("data/sample_docs")

    print("Creating vector store...")
    vector_store = create_vector_store(documents)

    print("Building RAG pipeline...")
    qa_chain = build_rag_chain(vector_store)

    queries = [
        "What is LangChain?",
        "Explain Azure AI Search",
        "What is RAG and why is it useful?"
    ]

    for query in queries:
        print("\nUser Query:", query)
        response = qa_chain.invoke({"query": query})
        print("Answer:", response["result"])

if __name__ == "__main__":
    main()
