from langchain_core.language_models import BaseChatModel
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_openai import ChatOpenAI

from demo_mlflow_agent_tracing.settings import Settings


def get_chat_model() -> BaseChatModel:
    """Get the chat model from environment: OpenAI (API key) or Claude on Vertex (project, region, model)."""
    settings = Settings()

    if settings.vertex_enabled:
        return ChatAnthropicVertex(
            project=settings.VERTEX_PROJECT_ID,
            location=settings.VERTEX_REGION,
            model_name=settings.VERTEX_MODEL_NAME,
        )
    # OpenAI-compatible with API key
    return ChatOpenAI(
        base_url=settings.OPENAI_BASE_URL,
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )
