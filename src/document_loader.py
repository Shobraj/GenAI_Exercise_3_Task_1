from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def load_documents(data_dir: str):
    documents = []

    for file_path in Path(data_dir).glob("*.txt"):
        loader = TextLoader(str(file_path))
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)
