"""Agent."""

import logging

import aiosqlite
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from demo_mlflow_agent_tracing.base import ContextSchema
from demo_mlflow_agent_tracing.chat_model import get_chat_model
from demo_mlflow_agent_tracing.constants import CHECKPOINTER_PATH, DIRECTORY_PATH
from demo_mlflow_agent_tracing.settings import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant. You answer questions using a knowledge base.

When a user asks a question, you must search for the answer in the knowledge base.

DO NOT provide any answer that is not supported by information from the knowledge base.

If you cannot find any information on the topic in the knowledge base, tell the user and do not attempt to answer the question on your own.
""".strip()


async def get_checkpointer_conn():
    """Get the database connection."""
    conn = await aiosqlite.connect(CHECKPOINTER_PATH)
    return conn


async def build_agent():
    """Build the agent."""
    # Construct the agent
    settings = Settings()
    llm = get_chat_model()
    llm.temperature = 0.0
    mcp_client = MultiServerMCPClient(
        {
            "content_writer": {
                "transport": "stdio",
                "command": "python",
                "args": [str(DIRECTORY_PATH / "src" / "demo_mlflow_agent_tracing" / "mcp_server.py")],
                "env": {
                    "OPENAI_API_KEY": settings.OPENAI_API_KEY.get_secret_value(),
                    "OPENAI_MODEL_NAME": settings.OPENAI_MODEL_NAME,
                    "OPENAI_BASE_URL": settings.OPENAI_BASE_URL,
                    "CHAINLIT_AUTH_SECRET": settings.CHAINLIT_AUTH_SECRET.get_secret_value(),
                    "EMBEDDING_API_KEY": settings.EMBEDDING_API_KEY.get_secret_value(),
                    "EMBEDDING_MODEL_NAME": settings.EMBEDDING_MODEL_NAME,
                    "EMBEDDING_BASE_URL": settings.EMBEDDING_BASE_URL,
                },
            }
        }
    )
    tools = await mcp_client.get_tools()
    conn = await get_checkpointer_conn()
    checkpointer = AsyncSqliteSaver(conn=conn)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        context_schema=ContextSchema,
        checkpointer=checkpointer,
    )

    return agent
