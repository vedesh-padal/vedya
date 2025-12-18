import sys
import json
import pandas as pd
from loguru import logger
from src.core.config import settings
from src.ingestion.loader import WhatsAppIngestor
from src.ingestion.transformer import ChatTransformer


def run_pipeline():
    logger.info("🚀 Starting Vedya Pipeline...")

    # --- Step 1: Ingestion ---
    logger.info("Step 1: Ingesting Chat Data")
    try:
        ingestor = WhatsAppIngestor(settings.CHAT_FILE_NAME)
        raw_messages = ingestor.load_messages()

        if not raw_messages:
            logger.error("No messages found. Exiting.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Ingestion Error: {e}")
        sys.exit(1)

    # --- Step 2: Transformation ---
    logger.info("Step 2: Transforming Data (Burge Merge & Chunking)")
    transformer = ChatTransformer()
    chunks = transformer.process(raw_messages)

    # --- Step 3: Saving (Parquest + JSON) ---
    # We save TWO files:
    # 1. Raw Messages (For Analytics)
    # 2. Chunks (For RAG / AI)

    # A. Save Raw Parquet (For Analytics)
    df_raw = ingestor.to_dataframe(raw_messages)
    raw_path = settings.PROCESSED_DATA_DIR / "chat_history_raw.parquet"
    df_raw.to_parquet(raw_path)
    logger.info(
        f"✅ Saved Raw Data (as .parquet file) ({len(df_raw)} rows) to: {raw_path}"
    )

    # B. Save Chunks Parquet (For RAG/AI) [ Convert Pydantic objects to Dicts ]
    # We use mode='python' to keep datetime objects for Parquet efficiency
    chunks_data_python = [c.model_dump(mode="python") for c in chunks]
    df_chunks = pd.DataFrame(chunks_data_python)

    # Drop 'messages' list for the Parquet file (it's too nested for flat tables)
    # We only need 'text_content' for the AI.
    df_chunks_simple = df_chunks.drop(columns=["messages"])
    chunk_path = settings.PROCESSED_DATA_DIR / "chat_history_chunks.parquet"
    df_chunks_simple.to_parquet(chunk_path)
    logger.success(f"✅ Saved {len(chunks)} Conversation Sessions (Chunks .parquet file) to: {chunk_path}")

    # C. Save Human-Readable JSON 
    # We use mode='json' to convert datetimes to Strings (ISO Format)
    chunks_data_json = [c.model_dump(mode='json') for c in chunks]

    json_path = settings.PROCESSED_DATA_DIR / "chat_history_chunks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data_json, f, indent=2, ensure_ascii=False)

    logger.success(f"✅ Saved Human-Readable JSON: {json_path}")

if __name__ == "__main__":
    run_pipeline()
