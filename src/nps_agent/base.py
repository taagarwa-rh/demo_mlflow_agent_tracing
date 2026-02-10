"""Base classes for all subagents."""

from operator import add
from typing import Annotated

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class State(BaseModel):
    """
    Graph state.

    Args:
        messages (list[BaseMessage]): Messages in the conversation.

    """

    messages: Annotated[list, add]

    @property
    def last_message(self) -> BaseMessage:
        """Get the most recent message."""
        return self.messages[-1]


class ContextSchema(BaseModel):
    """
    Schema for graph context.

    Args:
        user_info (str): User identifier string.

    """

    user_info: str
