from langchain_community.vectorstores.azuresearch import AzureSearch
from embedding import get_embedding_model
from config import (
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_KEY,
    AZURE_SEARCH_INDEX
)

def create_vector_store(documents):
    embedding_model = get_embedding_model()

    vector_store = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX,
        embedding_function=embedding_model.embed_query
    )

    vector_store.add_documents(documents)
    return vector_store
