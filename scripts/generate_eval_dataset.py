import mlflow
from demo_mlflow_agent_tracing.constants import DIRECTORY_PATH
from demo_mlflow_agent_tracing.settings import Settings
from mlflow.genai.datasets import create_dataset
from openai import OpenAI
from pydantic import BaseModel


class QuestionAnswerPair(BaseModel):
    """Represent a QnA pair."""

    index: int
    question: str
    answer: str


class QuestionAnswerPairs(BaseModel):
    """Request output format."""

    pairs: list[QuestionAnswerPair]


class MLFlowEvalData(BaseModel):
    """MLFlow eval data structure."""

    inputs: dict[str, str]
    expectations: dict[str, str]


PROMPT_TEMPLATE = """
You are an expert in AI system testing. Your task is to create question and answer pairs based on a given document.

Each question should be related to a relevant part of the document and should be clear and concise. Each answer should be a short, relevant sentence that directly answers the question.

Questions must not ask about document structure, headers, authors, metadata, or other such information. Questions should be focused on the content of the document and the information it is conveying.

Here is the document:

## START DOCUMENT ##

{document}

## END DOCUMENT ##

Please provide your {num_pairs} question and answer pairs about this document. Provide your response in JSON format as follows:

{{
    "pairs": [
        {{"index": 0, "question": "Question 1", "answer": "Answer 1"}},
        {{"index": 1, "question": "Question 2", "answer": "Answer 2"}},
        ...
    ]
}}

""".strip()


def sanitize_string(s: str) -> str:
    """Sanitize the results of the generation process."""
    sanitized = s
    sanitized = sanitized.replace("‑", "-").replace("’", "'").replace(" ", " ")
    return sanitized


def main():
    """Generate synthetic QnA pairs."""
    # Set up OpenAI Client
    settings = Settings()
    model = settings.OPENAI_MODEL_NAME
    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
    )

    # Fetch document paths
    directory = DIRECTORY_PATH / "public" / "oscorp_policies"
    paths = list(directory.glob("*.md"))

    # Generate QnA pairs
    num_pairs = 5
    records: list[MLFlowEvalData] = []
    for path in paths:
        # Prepare system prompt
        document = path.read_text()
        prompt = PROMPT_TEMPLATE.format(num_pairs=num_pairs, document=document)

        # Generate QnAs
        messages = [{"role": "user", "content": prompt}]
        response = client.beta.chat.completions.parse(
            messages=messages,
            response_format=QuestionAnswerPairs,
            model=model,
        )
        qna_pairs = response.choices[0].message.parsed

        # Save generation result in MLFlow format
        for pair in qna_pairs.pairs:
            inputs = {"question": sanitize_string(pair.question)}
            expectations = {"expected_response": sanitize_string(pair.answer), "expected_document": path.name}
            record = MLFlowEvalData(inputs=inputs, expectations=expectations)
            records.append(record.model_dump())

    # Save pairs to MLFlow
    if settings.MLFLOW_TRACKING_URI is not None:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    if settings.MLFLOW_EXPERIMENT_NAME is not None:
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    dataset = create_dataset(
        name="oscorp_policies_validation_set",
        tags={"stage": "validation", "environment": "dev", "model": model, "version": "0.1.0"},
    )
    dataset.merge_records(records=records)


if __name__ == "__main__":
    main()
