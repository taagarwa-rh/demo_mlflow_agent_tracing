import os
from typing import Literal

import mlflow
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolException, tool
from langchain_openai import ChatOpenAI


@tool
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
        raise ToolException("City must be 'SF' or 'NYC'")


@tool
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
        raise ToolException("Unit must be 'F' or 'C'")
    return converted


def main():
    """Run main process."""
    # Load environment variables, including OpenAI and MLflow variables
    # Two environment variables are required for autologging, MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME
    load_dotenv()

    # Instrument tracing - it's just one line!
    mlflow.langchain.autolog()

    # Create the OpenAI Client
    model_name = os.getenv("OPENAI_MODEL_NAME")
    model = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), model=model_name, temperature=0.1)

    # Create agent
    tools = [get_temperature, convert_temperature]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="Use the available tools to help answer the user's questions.",
    )

    # Process a request
    request = "What is the temperature in San Francisco in Farenheight?"
    print("Request:\n", request)
    input = {"messages": [{"role": "user", "content": request}]}
    response = agent.invoke(input=input)
    print("Response:\n", response.get("messages", [])[-1].content)


if __name__ == "__main__":
    main()
