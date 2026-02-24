import os

import mlflow
from dotenv import load_dotenv
from openai import OpenAI


def main():
    """Run main process."""
    # Load environment variables, including OpenAI and MLflow variables
    # Two environment variables are required for autologging, MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME
    load_dotenv()

    # Instrument tracing - it's just one line!
    mlflow.openai.autolog()

    # Create the OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    model = os.getenv("OPENAI_MODEL_NAME")

    # Trace a single message - no additional changes
    content = "What is the capital of France?"
    print("Request:\n", content)
    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(messages=messages, model=model).choices[0].message.content
    print("Response:\n", response)


if __name__ == "__main__":
    main()
