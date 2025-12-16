from langchain_openai import AzureOpenAIEmbeddings
from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    OPENAI_API_VERSION
)

def get_embedding_model():
    return AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=OPENAI_API_VERSION
    )
