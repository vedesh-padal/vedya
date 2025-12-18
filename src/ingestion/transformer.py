import uuid
from typing import List
from datetime import timedelta
from loguru import logger
from src.ingestion.schema import ChatMessage, ConversationChunk


class ChatTransformer:
    def __init__(self, burst_threshold_sec: int = 60, session_threshold_min: int = 20):
        """
        Args:
            burst_threshold_sec: If same user sends msg within 60s, merge them.
            session_threshold_min: If silence > 20 mins, start a new session.
        """
        self.burst_threshold = timedelta(seconds=burst_threshold_sec)
        self.session_threshold = timedelta(minutes=session_threshold_min)

    def process(self, messages: List[ChatMessage]) -> List[ConversationChunk]:
        """
        The Main Pipeline: Raw Messages -> Merged Bursts -> Conversation Chunks
        """
        if not messages:
            logger.warning("No messages to transform.")
            return []

        logger.info(f"Transforming {len(messages)} raw messages...")

        # 1. Merge Bursts (The "Spam" Handler)
        merged_messages = self._merge_bursts(messages)
        logger.info(f"Burst Messaging: Reduced to {len(merged_messages)} messages.")

        # 2. Create Sessions (The "Context" Handler)
        chunks = self._create_sessions(merged_messages)
        logger.success(
            f"Session Chunking: Created {len(chunks)} conversation messages."
        )

        return chunks

    def _merge_bursts(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """
        Combines consecutive messsages from the same sendder if close in time.
        """
        merged = []
        buffer = [messages[0]]

        for curr_msg in messages[1:]:
            prev_msg = buffer[-1]

            time_diff = curr_msg.timestamp - prev_msg.timestamp

            if (curr_msg.sender == prev_msg.sender) and (
                time_diff <= self.burst_threshold
            ):
                buffer.append(curr_msg)
            else:
                # Flush Buffer (Save the burst as one message)
                merged.append(self._flush_buffer(buffer))
                # Start new buffer
                buffer = [curr_msg]

        # Flush the final buffer
        if buffer:
            merged.append(self._flush_buffer(buffer))

        return merged

    def _flush_buffer(self, buffer: List[ChatMessage]) -> ChatMessage:
        """
        Helper to combine a list of messages into one.
        """
        combined_content = " ".join([m.content for m in buffer])

        # Use the timestamp of the FIRST message in the burst
        first_msg = buffer[0]

        return ChatMessage(
            timestamp=first_msg.timestamp,
            sender=first_msg.sender,
            content=combined_content,
            is_media="Media omitted" in combined_content,
        )

    def _create_sessions(self, messages: List[ChatMessage]) -> List[ConversationChunk]:
        """
        Group messages into sessions based on time gaps.
        """
        chunks = []
        current_session = [messages[0]]

        for i in range(1, len(messages)):
            prev_msg = messages[i - 1]
            curr_msg = messages[i]

            time_diff = curr_msg.timestamp - prev_msg.timestamp

            if time_diff > self.session_threshold:
                # Big Gap? Finish previous session.
                chunks.append(self._build_chunk(current_session))
                current_session = []

            current_session.append(curr_msg)

        # Finish final session
        if current_session:
            chunks.append(self._build_chunk(current_session))

        return chunks

    def _build_chunk(self, msgs: List[ChatMessage]) -> ConversationChunk:
        """
        Converts a list of messages into a ConversationChunk object.
        """
        # Formats text for the AI to read later
        # formatted_text = "\n".join([f"{m.sender}: {m.content}" for m in msgs])

        # for a neater representation for the AI, updating text_content, so that AI sees time
        formatted_lines = []
        for m in msgs:
            readable_time = m.timestamp.strftime("%I:%M %p")
            line = f"[{readable_time}] {m.sender}: {m.content}"
            formatted_lines.append(line)

        formatted_text = "\n".join(formatted_lines)
        participants = list(set(m.sender for m in msgs))

        return ConversationChunk(
            conversation_id=str(
                uuid.uuid4()
            ),  # UUIDs prevent conflicts in distributed systems by ensuring globally unique identifiers, unlike serial numbers which can cause collisions during merges.
            date=msgs[0].timestamp.strftime("%Y-%m-%d"),
            start_time=msgs[0].timestamp,
            end_time=msgs[-1].timestamp,
            participants=participants,
            text_content=formatted_text,
            messages=msgs,
        )
