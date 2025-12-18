import re
import pandas as pd
from typing import List
from loguru import logger
from src.core.config import settings
from src.ingestion.schema import ChatMessage
from datetime import datetime

class WhatsAppIngestor:
    def __init__(self, file_path: str):
        self.file_path = settings.RAW_DATA_DIR / file_path
        self.pattern = re.compile(settings.TIMESTAMP_REGEX)

    def _parse_line(self, line: str):
        """Helper to check if a line starts with a timestamp."""
        match = self.pattern.match(line)
        if match:
            return match.groups(), line[match.end():]
        return None, line
    
    def load_messages(self) -> List[ChatMessage]:
        """
        Parses the raw text file into validated Pydantic models.
        """
        logger.info(f"Starting ingestion from {self.file_path}")

        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"Please put {settings.CHAT_FILE_NAME} in data/raw/")
        
        parsed_messages = []

        # State variables for multi-line handling
        current_date_str = None
        current_time_str = None
        current_sender = None
        message_buffer = []

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip strict empty lines, but keep lines that might be just emojis
                if not line:
                    continue

                timestamp_parts, remaining_text = self._parse_line(line)

                if timestamp_parts and timestamp_parts[0]: # It's a new message
                    # 1. Flush previous message
                    if message_buffer and current_sender:
                        self._create_and_append(
                            parsed_messages,
                            current_date_str,
                            current_time_str,
                            current_sender,
                            message_buffer
                        )
                    
                    # 2. Start new message
                    current_date_str, current_time_str = timestamp_parts
                    message_buffer = []

                    # Extract sender
                    if ": " in remaining_text:
                        current_sender, content = remaining_text.split(": ", 1)
                        message_buffer.append(content)
                    else:
                        # System message (e.g., "Messages are encrypted")
                        current_sender = "System"
                        message_buffer.append(remaining_text)
                        # The best way is to ignore this message, but just proceeding 
                        # to keep it, as it may be the least useful analytic

                else:
                    # Continuation of previous message
                    message_buffer.append(line)

            # Flush the final message
            if message_buffer and current_sender:
                self._create_and_append(
                    parsed_messages,
                    current_date_str,
                    current_time_str,
                    current_sender,
                    message_buffer
                )
        
        logger.success(f"Ingested {len(parsed_messages)} messages successfully.")
        return parsed_messages
    
    def _create_and_append(self, list_ref, date, time, sender, buffer):
        """Helper to validate and append a message."""
        full_content = " ".join(buffer)

        # Filter out system messages immediately (Optional)
        if sender == "System":
            return
        
        try:
            # Convert string to datetime
            dt_obj = datetime.strptime(
                f"{date} {time}", 
                settings.DATE_FORMAT
            )

            # Create Pydantic Model (Validation happens here)
            msg = ChatMessage(
                timestamp=dt_obj,
                sender=sender,
                content=full_content,
                is_media="<Media omitted>" in full_content
            )
            list_ref.append(msg)

        except ValueError as e:
            logger.warning(f"Failed to parse date: {date} {time} - {e}")

    def to_dataframe(self, messages: List[ChatMessage]) -> pd.DataFrame:
        """Converts Pydantic models to Pandas DataFrame for analytics."""
        # model_dump() is the Pydantic v2 method (dict() is deprecated)
        data = [msg.model_dump() for msg in messages]
        return pd.DataFrame(data)
    

if __name__ == "__main__":
    # Test run
    ingestor = WhatsAppIngestor(settings.CHAT_FILE_NAME)
    msgs = ingestor.load_messages()
    df = ingestor.to_dataframe(msgs)

    # Save to Parquest (Better than CSV for preserving data types)
    output_path = settings.PROCESSED_DATA_DIR / "chat_history.parquet"
    df.to_parquet(output_path)
    logger.info(f"Saved processed data to {output_path}")