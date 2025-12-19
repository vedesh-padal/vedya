# Vedya 🤖

**Vedya** (Vedesh + Divya) is a local AI Agent and Analytics Engine designed to ingest, analyze, and "remember" years of WhatsApp chat history. 

It goes beyond simple keyword search by using **RAG (Retrieval Augmented Generation)** to understand the context of conversations and an **Insights Engine** to derive behavioral and emotional metrics from the chat.

> **Privacy Note:** This project runs 100% locally. No chat data is sent to the cloud.

## 🚀 Features

### Phase 1: Data Engineering (Complete)
- **Robust Ingestion:** Parses raw WhatsApp `.txt` exports, handling multi-line messages and system events.
- **Burst Merging:** Intelligently merges consecutive short messages (spam/bursts) into coherent blocks.
- **Session Chunking:** Groups messages into "Conversation Sessions" based on time gaps for better AI context.

### Phase 2: Insights Engine (Complete)
- **Deep Analytics:** Calculates response latency, initiation rates, and double-texting frequency.
- **Linguistic Analysis:** Extracts top emojis, vocabulary size, and question ratios.
- **Sentiment Analysis:** Uses `XLM-RoBERTa` (Multilingual) to detect emotional tone in mixed English/Telugu/Hindi text.
- **Profile Generation:** Creates a `relationship_profile.json` that summarizes the friendship's "Vibe."

### Phase 3 & 4: RAG & Agent (In Progress)
- **Vector Memory:** Storing chat chunks in ChromaDB for semantic search.
- **Chat Interface:** A Streamlit UI to talk to your chat history.

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Package Manager:** `uv` (The modern replacement for pip/poetry)
- **Data Processing:** Pandas, Pydantic
- **AI/NLP:** PyTorch, Transformers (HuggingFace), Sentence-Transformers
- **Vector DB:** ChromaDB
- **Logging:** Loguru

## 📂 Project Structure

```text
vedya/
├── data/
│   ├── raw/            # Put your chat.txt here (Gitignored)
│   └── processed/      # Generated Parquet/JSON files
├── src/
│   ├── analytics/      # Sentiment & Stats Logic
│   ├── core/           # Config & Settings
│   ├── ingestion/      # Parsing & Transformation Logic
│   └── rag/            # Vector Database Logic
├── main.py             # The Orchestrator Entry Point
├── pyproject.toml      # Dependencies
└── README.md
```

## ⚡ Getting Started

### Prerequisites

  - Python 3.11+
  - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)

### Installation

1. **Clone the repository**

    ```sh    
    git clone https://github.com/yourusername/vedya.git
    cd vedya
    ```

2. **Install Dependencies**

    uv will automatically create the virtual environment and install packages.
    ```sh      
    uv sync
    ```

3. **Setup Data**

   - Export your WhatsApp chat (without media).
   - Rename the file to chat.txt (or update src/core/config.py).
   - Place it in data/raw/.

### Running the Pipeline

To run the Ingestion, Transformation, and Analytics phases:
```sh  
uv run python main.py
```