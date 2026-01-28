"""Main application for the expense agent."""

import json
import logging
from typing import Any, AsyncIterator, Union

import chainlit as cl
import mlflow
from langchain_core.messages import AIMessageChunk
from langgraph.types import Command
from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

from demo_mlflow_agent_tracing.agent import build_agent
from demo_mlflow_agent_tracing.base import ContextSchema
from demo_mlflow_agent_tracing.settings import Settings

# Validate settings
settings = Settings()

# Start logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# Log settings
logger.info(f"Settings loaded: {settings}")

# Start MLFlow Autolog
mlflow.langchain.autolog()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authorize using basic auth."""
    # Allow in any users with the right password, but capture their username
    if password == "admin":
        return cl.User(identifier=username, metadata={"role": "admin", "provider": "credentials"})
    else:
        return None


@cl.set_starters
async def set_starters():
    """Set conversation starters."""
    return [
        cl.Starter(
            label="Create",
            message="Please create an article about aurora borealis.",
            icon="/public/sparkle.svg",
        ),
        cl.Starter(
            label="List",
            message="List all available articles.",
            icon="/public/list-bullet.svg",
        ),
        cl.Starter(
            label="Search",
            message="Find me articles about animals.",
            icon="/public/magnifying-glass.svg",
        ),
    ]


@cl.on_chat_start
async def start_chat():
    """Run on chat start."""
    cl.user_session.set("message_history", [])


@cl.on_settings_update
async def setup_chat(chat_settings: dict[str, Any]):
    """Apply chat settings."""
    cl.user_session.set("chat_settings", chat_settings)


async def thinking(generator: AsyncIterator[dict[str, Any] | Any]):
    """Run thinking step."""
    async with cl.Step(name="Thinking", type="llm") as thinking_step:
        async for token, metadata in generator:
            if token.content == "</think>":
                break
            await thinking_step.stream_token(token.content)


async def tool_response(token):
    """Handle tool response."""
    if token.content:
        async with cl.Step(name="Tool Response") as tool_response_step:
            tool_response_step.output = f"```\n{token.content}\n```"


async def tool_call(generator: AsyncIterator[dict[str, Any] | Any], init_token: AIMessageChunk):
    """Handle tool call (which may appear in chunks)."""
    # If the tool calls arrive in chunks, iterate through and combine the chunks
    if hasattr(init_token, "tool_call_chunks") and init_token.tool_call_chunks:
        tool_calls = init_token
        async for token, _ in generator:
            if hasattr(token, "tool_call_chunks") and token.tool_call_chunks:
                tool_calls += token
            else:
                tool_calls = tool_calls.tool_call_chunks
                break

    # If the tool calls appear as one object, simply grab the object
    elif hasattr(init_token, "tool_calls") and init_token.tool_calls:
        tool_calls = init_token.tool_calls

    # If there's anything else, raise an error
    else:
        raise AttributeError("Token is missing tool call attributes")

    # Write the tool call to a tool call step
    async with cl.Step(name="Tool Call") as tool_call_step:
        output = [
            {
                "name": tool["name"],
                "args": (json.loads(tool["args"]) if isinstance(tool["args"], str) else tool["args"]),
            }
            for tool in tool_calls
        ]
        output = json.dumps(output, indent=2)
        tool_call_step.output = f"```json\n{output}\n```"


async def stream_agent_response(input: Union[str, Command], config: dict, context: ContextSchema, message_history: list[dict[str, Any]]):
    """Stream the agent response to chat messages."""
    last_node = ""
    msg = None
    # Create the response generator
    agent = await build_agent()
    generator = agent.astream(input=input, config=config, context=context, stream_mode="messages")
    async for token, metadata in generator:
        # Start a new message if the node has changed
        node = metadata["langgraph_node"]
        if node != last_node:
            if msg is not None:
                await msg.update()
                message_history.append({"role": "assistant", "content": msg.content})
            msg = None

        # Put the agent content in a message
        if token.content:
            # If we're thinking, wrap it in the collapsible "Using Thinking" step
            if token.content == "<think>":
                await thinking(generator)

            # If we're getting a tool response, wrap it in a tool response step
            elif "tools" in node:
                await tool_response(token=token)

            # Otherwise, just put it in a normal message
            else:
                if msg is None:
                    msg = cl.Message(content="")
                await msg.stream_token(token.content)

        if (hasattr(token, "tool_calls") and token.tool_calls) or (hasattr(token, "tool_call_chunks") and token.tool_call_chunks):
            await tool_call(generator=generator, init_token=token)

        last_node = node

    # Close out the message stream
    if msg is not None:
        await msg.update()
        message_history.append({"role": "assistant", "content": msg.content})


def get_app_user() -> cl.User:
    """Get the active user info."""
    app_user = cl.user_session.get("user")
    logger.info(f"User information: {app_user}")

    return app_user


@cl.on_message
async def main(message: cl.Message):
    """Run main messaging process."""
    # Get the message history
    message_history = cl.user_session.get("message_history")
    messages = [{"role": "user", "content": message.content}]
    message_history.extend(messages)

    # Get the active user and add to trace
    app_user = get_app_user()
    mlflow.update_current_trace(metadata={"mlflow.trace.user": app_user.identifier})

    # Construct the graph input
    input = {"messages": messages, "user_info": app_user.identifier}
    config = {"configurable": {"thread_id": message.thread_id}, "callbacks": [MlflowLangchainTracer()]}

    # Set user context
    context = ContextSchema(user_info=app_user.identifier)

    # Stream the agent response
    await stream_agent_response(input=input, config=config, context=context, message_history=message_history)
