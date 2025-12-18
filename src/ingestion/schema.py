from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class ChatMessage(BaseModel):
    """
    Represents a single atomic message in the system
    """
    timestamp: datetime
    sender: str
    content: str

    # Metadat fields (we will populate these later)
    is_media: bool = False
    sentiment_score: Optional[float] = None

    class Config:
        frozen = True # Makes instances immutable (Thread-safe)