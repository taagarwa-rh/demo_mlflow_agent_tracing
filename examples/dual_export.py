import mlflow
import os
import openai

# Enable dual-exporting: https://mlflow.org/docs/latest/genai/tracing/opentelemetry/export/#dual-export
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT"] = "true"
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:4317/v1/traces"

# Start autologging
mlflow.openai.autolog()

# Run the OpenAI chat completion
prompt = "What is the capital of France?"
client = openai.OpenAI(api_key="NONE", base_url="http://localhost:11434/v1")
response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[{"role": "user", "content": prompt}],
)
print(response.choices[0].message.content)
