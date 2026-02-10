"""Agent."""

import logging

import aiosqlite
import mlflow
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.runtime import Runtime
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

from nps_agent.base import ContextSchema
from nps_agent.chat_model import get_chat_model
from nps_agent.constants import CHECKPOINTER_PATH, DIRECTORY_PATH
from nps_agent.settings import Settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant. You help users get information about the United States National Parks Service (NPS). Use the available tools to answer the user's questions. 

Most tools require the park code in order to get information. Use the search_parks tool first to find the park code, then use it in other tool calls.

IMPORTANT: Do not attempt to answer the question on your own, use the available tools to help answer the questions.

IMPORTANT: Do not answer any questions not related to National Parks. Explain that you cannot help with those requests but are happy to help with NPS requests.
""".strip()


async def get_checkpointer_conn():
    """Get the database connection."""
    conn = await aiosqlite.connect(CHECKPOINTER_PATH)
    return conn


def format_input(content: str, user_identifier: str):
    """Format graph input."""
    messages = [{"role": "user", "content": content}]
    input = {"messages": messages, "user_info": user_identifier}
    return input


def format_config(thread_id: str):
    """Format graph config."""
    config = {"configurable": {"thread_id": thread_id}, "callbacks": [MlflowLangchainTracer(run_inline=True)]}
    return config


def format_context(user_identifier: str):
    """Format graph context."""
    context = ContextSchema(user_info=user_identifier)
    return context


@before_agent
def update_tracing(state: AgentState, runtime: Runtime):
    """Update MLFlow tracing params (no-op when no active trace, e.g. during eval)."""
    context: ContextSchema = runtime.context
    user = context.user_info
    try:
        mlflow.update_current_trace(metadata={"mlflow.trace.user": user})
    except Exception:
        # No active trace or non-recording span (e.g. during mlflow.genai.evaluate)
        pass


async def build_agent(*, return_connection: bool = False, use_memory_checkpointer: bool = False):
    """
    Build the agent.

    Args:
        use_memory_checkpointer (bool): use InMemorySaver (no DB); for evals so traces run on same thread and no aiosqlite.
        return_connection (bool): when False and not use_memory_checkpointer, returns (agent, conn) for caller to close conn.

    """
    # Construct the agent
    settings = Settings()

    # Get the chat model
    llm = get_chat_model()
    llm.temperature = 0.7

    # Build MCP env: include OpenAI or Vertex vars depending on which LLM backend is configured
    mcp_env = {
        "NPS_API_KEY": settings.NPS_API_KEY.get_secret_value(),
        "CHAINLIT_AUTH_SECRET": (settings.CHAINLIT_AUTH_SECRET.get_secret_value() if settings.CHAINLIT_AUTH_SECRET else ""),
    }
    if settings.openai_enabled:
        mcp_env["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()
        mcp_env["OPENAI_MODEL_NAME"] = settings.OPENAI_MODEL_NAME
        mcp_env["OPENAI_BASE_URL"] = settings.OPENAI_BASE_URL or ""
    if settings.vertex_enabled:
        mcp_env["VERTEX_PROJECT_ID"] = settings.VERTEX_PROJECT_ID
        mcp_env["VERTEX_REGION"] = settings.VERTEX_REGION
        mcp_env["VERTEX_MODEL_NAME"] = settings.VERTEX_MODEL_NAME

    # Get tools from MCP server
    mcp_client = MultiServerMCPClient(
        {
            "content_writer": {
                "transport": "stdio",
                "command": "python",
                "args": [str(DIRECTORY_PATH / "src" / "nps_agent" / "mcp_server.py")],
                "env": mcp_env,
            }
        }
    )
    tools = await mcp_client.get_tools()

    # Load system prompt from MLFlow if requested
    if settings.MLFLOW_SYSTEM_PROMPT_URI is not None:
        logger.info(f"Loading prompt from MLFlow: {settings.MLFLOW_SYSTEM_PROMPT_URI}")
        system_prompt = mlflow.genai.load_prompt(settings.MLFLOW_SYSTEM_PROMPT_URI).format()
    else:
        logger.info("No system prompt specified. Using default system prompt.")
        system_prompt = SYSTEM_PROMPT

    # Create agent (in-memory checkpointer for evals to avoid aiosqlite + preserve trace context)
    if use_memory_checkpointer:
        checkpointer = InMemorySaver()
    else:
        conn = await get_checkpointer_conn()
        checkpointer = AsyncSqliteSaver(conn=conn)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        context_schema=ContextSchema,
        checkpointer=checkpointer,
        middleware=[update_tracing],
    )

    if not use_memory_checkpointer and return_connection:
        return (agent, conn)
    return agent
