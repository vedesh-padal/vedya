import requests
from loguru import logger

MODEL = settings.MODEL
OLLAMA_URL = settings.OLLAMA_URL

def ollama_llm(prompt: str) -> str:
    logger.info("Inside Ollama LLM wrapper")
    try:
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=600)
        response.raise_for_status()

        return response.json().get("response", "").strip()

    except Exception as e:
        logger.error(f"Ollama LLM call failed: {e}")
        return ""
