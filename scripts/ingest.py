import logging

from datasets import load_dataset
from demo_mlflow_agent_tracing.constants import DB_PATH
from demo_mlflow_agent_tracing.db import get_db
from langchain_core.documents import Document
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)


def main():
    """Run ingestion."""
    # Check if the Chroma store exists already
    if (DB_PATH / "chroma.sqlite3").exists():
        logger.info("Chroma store already exists, skipping creation.")
        return

    logger.info("Chroma store does not exist, creating...")

    # Load the text corpus
    dataset = "rag-datasets/rag-mini-wikipedia"
    corpus = load_dataset(dataset, "text-corpus")["passages"]
    logger.info(f"Read {corpus.num_rows} texts from {dataset}")

    # Set up Chroma DB
    db = get_db()
    db.reset_collection()

    # Load all texts as documents
    for row in tqdm(corpus, desc="Embedding documents..."):
        # Get the file content and id
        content = row["passage"]
        row_id = row["id"]

        # Create a document
        document = Document(page_content=content, metadata={"row_id": row_id, "dataset": dataset})

        # Save the document to chroma
        db.add_documents(documents=[document])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
