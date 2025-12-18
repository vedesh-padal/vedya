from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class ChatMessage(BaseModel):
    """
    Represents a single atomic message
    """
    timestamp: datetime
    sender: str
    content: str

    # Metadata fields (we will populate these later)
    is_media: bool = False
    # sentiment_score: Optional[float] = None

    class Config:
        frozen = True # Makes instances immutable (Thread-safe)

class ConversationChunk(BaseModel):
    """
    Represents a 'Session' or 'Block' of conversation.
    This is the unit we will eventually feed to the AI (RAG).
    """
    conversation_id: str = Field(..., description="Unique UUID for this session")
    date: str # YYYY-MM-DD
    start_time: datetime
    end_time: datetime
    participants: List[str]

    # The formatted text the AI will read:
    # "Vedesh: Hi \n Divya: Hello"
    text_content: str

    # We keep the raw objects too, just in case
    messages: List[ChatMessage]