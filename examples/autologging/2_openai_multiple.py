import os
from uuid import uuid4

import mlflow
from dotenv import load_dotenv
from openai import OpenAI


@mlflow.trace
def get_chat_completion(messages: list[dict], session_id: str = None, user_id: str = None):
    """Get a chat completion."""
    # Add session details
    mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id, "mlflow.trace.user": user_id})

    # Create the OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    model = os.getenv("OPENAI_MODEL_NAME")

    # Create chat completion
    response = client.chat.completions.create(messages=messages, model=model)
    return response.choices[0].message.content


def main():
    """Run main process."""
    # Load environment variables, including OpenAI and MLflow variables
    # Two environment variables are required for autologging, MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME
    load_dotenv()

    # Instrument tracing - it's just one line!
    mlflow.openai.autolog()

    # Trace multiple messages in a session - requires an mlflow.trace decorated function
    prompt1 = "Write a short story about a man who lives in a shoe."
    prompt2 = "Please summarize your story in two sentences."
    session_id = str(uuid4())
    user_id = "demo"
    messages = [{"role": "user", "content": prompt1}]
    story = get_chat_completion(messages=messages, session_id=session_id, user_id=user_id)
    messages.append({"role": "assistant", "content": story})
    messages.append({"role": "user", "content": prompt2})
    summary = get_chat_completion(messages=messages, session_id=session_id, user_id=user_id)
    print("Prompt:\n", prompt1)
    print("Story:\n", story)
    print("Summary:\n", summary)


if __name__ == "__main__":
    main()
