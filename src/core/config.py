from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # File Names
    CHAT_FILE_NAME: str = "chat.txt"

    # Parsing Rules (Regex for 26/11/25, 6:23 PM - )
    TIMESTAMP_REGEX: str = r'^(\d{2}/\d{2}/\d{2}),\s(\d{1,2}:\d{2}\s(?:am|pm))\s-\s'
    DATE_FORMAT: str = "%d/%m/%y %I:%M %p"
    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL = "llama3.2:3b"


    class Config:
        env_file = ".env"

settings = Settings()