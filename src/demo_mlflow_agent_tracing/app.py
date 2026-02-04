"""Main application for the expense agent."""

import json
import logging
from typing import Any, AsyncIterator

import chainlit as cl
from langchain_core.messages import AIMessageChunk

from demo_mlflow_agent_tracing.agent import build_agent, format_config, format_context, format_input
from demo_mlflow_agent_tracing.settings import Settings

# Validate settings
settings = Settings()

# Start logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# Log settings
logger.info(f"Settings loaded: {settings}")


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
            label="Ask",
            message="What is included in the Dark-Side Health Plan?",
            icon="/public/sparkle.svg",
        ),
        cl.Starter(
            label="Irrelevant",
            message="What is the airspeed velocity of an unladen swallow?",
            icon="/public/x-circle.svg",
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


async def stream_agent_response(message: cl.Message, app_user: cl.User):
    """Stream the agent response to chat messages."""
    # Create graph input
    input = format_input(content=message.content, user_identifier=app_user.identifier)
    config = format_config(thread_id=message.thread_id)
    context = format_context(user_identifier=app_user.identifier)

    # Initialize loop variables
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
            msg = None

        # Put the agent content in a message
        if token.content:
            # If we're getting a tool response, wrap it in a tool response step
            if "tools" in node:
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


def get_app_user() -> cl.User:
    """Get the active user info."""
    app_user = cl.user_session.get("user")
    logger.info(f"User information: {app_user}")

    return app_user


@cl.on_message
async def main(message: cl.Message):
    """Run main messaging process."""
    # Get the active user and add to trace
    app_user = get_app_user()

    # Stream the agent response
    await stream_agent_response(message=message, app_user=app_user)
