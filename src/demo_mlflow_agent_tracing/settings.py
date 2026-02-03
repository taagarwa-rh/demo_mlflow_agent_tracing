"""Configuration settings for the expense agent."""

from typing import Literal, Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow", env_ignore_empty=True)

    # LLM provider: "openai" (API key) or "vertex" (Claude on Vertex)
    LLM_PROVIDER: Literal["openai", "vertex"] = Field(
        "openai",
        description="Which LLM backend to use: 'openai' or 'vertex'.",
    )

    # OpenAI (for OpenAI-compatible API with API key)
    OPENAI_API_KEY: Optional[SecretStr] = Field(None, description="API key for authenticating with the server")
    OPENAI_MODEL_NAME: Optional[str] = Field(None, description="Name of the model to use (e.g. `qwen3:8b`)")
    OPENAI_BASE_URL: Optional[str] = Field(None, description="Base URL of the server")

    # Vertex AI (for Claude on Vertex: project ID, region, model name)
    VERTEX_PROJECT_ID: Optional[str] = Field(None, description="GCP project ID for Vertex AI")
    VERTEX_REGION: Optional[str] = Field(None, description="Vertex AI region (e.g. us-central1)")
    VERTEX_MODEL_NAME: Optional[str] = Field(None, description="Claude model name on Vertex (e.g. claude-3-5-sonnet@20241022)")

    # Embedding Server
    EMBEDDING_API_KEY: Optional[SecretStr] = Field(None, description="API key for authenticating with the server")
    EMBEDDING_MODEL_NAME: Optional[str] = Field(None, description="Name of the model to use (e.g. `nomic-embed-text`)")
    EMBEDDING_BASE_URL: Optional[str] = Field(None, description="Base URL of the server")
    EMBEDDING_DOCUMENT_PREFIX: Optional[str] = Field("", description="Prefix for embeddings for documents")
    EMBEDDING_SEARCH_PREFIX: Optional[str] = Field("", description="Prefix for embeddings for search queries")

    # Chainlit
    CHAINLIT_AUTH_SECRET: Optional[SecretStr] = Field(
        None,
        description="Authorization secret used for signing tokens. Can be generated using `chainlit create-secret`",
    )

    # MLFlow
    MLFLOW_TRACKING_URI: Optional[str] = Field(None, description="MLFlow Tracking URI")
    MLFLOW_EXPERIMENT_NAME: Optional[str] = Field(None, description="MLFlow Experiment Name")
    MLFLOW_SYSTEM_PROMPT_URI: Optional[str] = Field(None, description="MLFlow Prompt URI (e.g. prompts:/my-prompt@latest)")

    @property
    def openai_enabled(self) -> bool:
        """True when LLM_PROVIDER is 'openai'."""
        return self.LLM_PROVIDER == "openai"

    @property
    def vertex_enabled(self) -> bool:
        """True when LLM_PROVIDER is 'vertex'."""
        return self.LLM_PROVIDER == "vertex"

    @property
    def auth_enabled(self) -> bool:
        """Check if required Keycloak environment variables are set."""
        return self.CHAINLIT_AUTH_SECRET is not None

    @property
    def embedding_server_enabled(self) -> bool:
        """Check if optional embedding server environment variables are set."""
        return self.EMBEDDING_API_KEY is not None and self.EMBEDDING_MODEL_NAME is not None

    @model_validator(mode="after")
    def llm(self) -> Self:
        """Validate that required environment variables are set for the selected LLM_PROVIDER."""
        if self.LLM_PROVIDER == "openai":
            if self.OPENAI_API_KEY is None or self.OPENAI_MODEL_NAME is None:
                raise Exception(
                    "LLM_PROVIDER is 'openai'. Set OPENAI_API_KEY and OPENAI_MODEL_NAME."
                )
        else:
            if (
                self.VERTEX_PROJECT_ID is None
                or self.VERTEX_REGION is None
                or self.VERTEX_MODEL_NAME is None
            ):
                raise Exception(
                    "LLM_PROVIDER is 'vertex'. Set VERTEX_PROJECT_ID, VERTEX_REGION, and VERTEX_MODEL_NAME."
                )
        return self

    # CHAINLIT_AUTH_SECRET is optional; required only when running the Chainlit chat UI.
