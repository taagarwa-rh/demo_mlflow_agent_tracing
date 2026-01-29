from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from demo_mlflow_agent_tracing.constants import DB_PATH
from demo_mlflow_agent_tracing.settings import Settings


def get_db() -> Chroma:
    """Get vector db."""
    # Get embedding function
    embedding_function = None
    settings = Settings()
    if settings.embedding_server_enabled:
        embedding_function = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
        )

    # Launch DB
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    return db
