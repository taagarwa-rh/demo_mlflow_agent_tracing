uv run --no-sync --locked scripts/ingest.py
uv run --no-sync --locked chainlit run src/demo_mlflow_agent_tracing/app.py --host 0.0.0.0 --port 8000