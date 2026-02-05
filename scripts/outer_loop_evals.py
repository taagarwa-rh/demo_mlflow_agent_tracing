import logging
import os
from typing import Any, Literal

import mlflow
from demo_mlflow_agent_tracing.mcp_server import SearchResult
from demo_mlflow_agent_tracing.settings import Settings
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.genai.judges import make_judge
from mlflow.entities import Feedback
from mlflow.genai import evaluate

logger = logging.getLogger(__name__)

def main() -> None:
    """Search production traces and run GenAI evaluation on them."""
    load_dotenv()
    settings = Settings()
    if settings.MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    if settings.MLFLOW_EXPERIMENT_NAME:
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    if settings.vertex_enabled:
        os.environ["VERTEXAI_PROJECT"] = settings.VERTEX_PROJECT_ID
        os.environ["VERTEXAI_LOCATION"] = settings.VERTEX_REGION

    client = MlflowClient()
    exp_name = settings.MLFLOW_EXPERIMENT_NAME or "Default"
    experiment = client.get_experiment_by_name(exp_name)
    if not experiment:
        raise SystemExit(
            f"Experiment '{exp_name}' not found. Create it or set MLFLOW_EXPERIMENT_NAME."
        )
    experiment_id = experiment.experiment_id

    max_traces = 5
    filter_string = "trace.status = 'OK'"

    logger.info("Searching traces: experiment_id=%s, filter=%s, max_results=%s", experiment_id, filter_string, max_traces)
    traces_df = mlflow.search_traces(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=max_traces,
        order_by=["trace.timestamp_ms DESC"],
    )

    if traces_df is None or traces_df.empty:
        logger.warning("No traces found. Run the agent (e.g. chainlit) and generate some traffic, then re-run.")
        return

    logger.info("Evaluating %s traces", len(traces_df))

    if settings.openai_enabled:
        model = f"openai:/{settings.OPENAI_MODEL_NAME}"
    else:
        model = f"vertex_ai:/{settings.VERTEX_MODEL_NAME}"
    
    # LLM as a judge
    grounded_judge = make_judge(
    name="groundedness",
    instructions=(
        "Verify the outputs are grounded in the context provided in the inputs and intermediate context from tool calls. {{ trace }}\n"
        "Rate: 'fully', 'partially', or 'not' grounded."
    ),
    feedback_value_type=Literal["fully", "partially", "not"],
    model=model,
    )

    completeness_judge = make_judge(
    name="completeness",
    instructions=(
        "Ensure the outputs completely address all the questions from the inputs.\n"
        "Inputs: {{ inputs }} \n Outputs: {{ outputs }} \n"
        "Rate as 'complete' or 'incomplete'."
    ),
    feedback_value_type=Literal["complete", "incomplete"],
    model=model,
    )
    # Agent as a judge
    error_handling_judge = make_judge(
        name="error_handler_checker",
        instructions=(
            "Analyze error handling in the {{ trace }}.\n\n"
            "Look for:\n"
            "1. Spans with error status or exceptions\n"
            "2. Retry attempts and their patterns\n"
            "3. Fallback mechanisms\n"
            "4. Error propagation and recovery\n\n"
            "Identify specific error scenarios and how they were handled.\n"
            "Rate as: 'robust', 'adequate', or 'fragile'"
        ),
        feedback_value_type=Literal["robust", "adequate", "fragile"],
        model=model,
    )

    scorers_list = [
        grounded_judge,
        completeness_judge,
        error_handling_judge,
    ]

    results = evaluate(data=traces_df, scorers=scorers_list)
    logger.info("Evaluation results: %s", results.metrics)
    if "eval_results_table" in results.tables:
        logger.info("Eval results table:\n%s", results.tables["eval_results_table"].to_string())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
