import os
from typing import Literal

import mlflow
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def get_temperature(city: Literal["SF", "NYC"]) -> str:
    """
    Get the current temperature in Celcius for a city.

    Args:
        city (Literal["SF", "NYC"]): City abbreviation, either NYC (New York City) or SF (San Francisco)

    Returns:
        str: Message containing the current temperature in the city in Celcius

    """
    if city == "SF":
        return f"It is currently 25°C in {city}"
    elif city == "NYC":
        return f"It is currently 10°C in {city}"
    else:
        raise ValueError("City must be 'SF' or 'NYC'")


def convert_temperature(value: float, unit: Literal["F", "C"]) -> float:
    """
    Convert temperature from one unit (F or C) to the opposite unit.

    Args:
        value (float): Temperature value
        unit (Literal["F", "C"]): Unit that value is in

    Returns:
        float: The converted temperature

    """
    if unit == "C":
        converted = value * 9 / 5 + 32
    elif unit == "F":
        converted = (value - 32) * 5 / 9
    else:
        raise ValueError("Unit must be 'F' or 'C'")
    return converted


class OutputType(BaseModel):
    """Agent output format."""

    city: str
    temperature: float
    unit: Literal["F", "C"]


def main():
    """Run main process."""
    # Load environment variables, including OpenAI and MLflow variables
    # Two environment variables are required for autologging, MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME
    load_dotenv()

    # Instrument tracing - it's just one line!
    mlflow.pydantic_ai.autolog()

    # Create the OpenAI Client
    model_name = os.getenv("OPENAI_MODEL_NAME")
    provider = OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    model = OpenAIChatModel(model_name=model_name, provider=provider)

    # Create agent
    tools = [get_temperature, convert_temperature]
    agent = Agent(
        model=model,
        tools=tools,
        instructions="Use the available tools to help answer the user's questions.",
        output_type=OutputType,
    )

    # Process a request - supports sync, async, and streaming
    request = "What is the temperature in San Francisco in Farenheight?"
    print("Request:\n", request)
    response = agent.run_sync(request)
    print("Response:\n", response.output)


if __name__ == "__main__":
    main()
