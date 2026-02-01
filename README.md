# Asar AI CV Assistant

A minimal, privacy-respecting AI chatbot that answers questions about my professional background using **only my CV as a source of truth**.

Built with:
- Python 3.12
- LangChain
- Google Gemini (chat + embeddings)
- ChromaDB
- Local vector storage

This project is designed as a **personal AI brand assistant**, not a generic chatbot.

---

## What this does

- Answers questions about my background, experience, and projects
- Uses **retrieval-augmented generation (RAG)** over my CV
- Enforces strict factual grounding (no hallucinated jobs, dates, or schools)
- Maintains short conversational memory (last ~10 messages)
- Responds in a calm, minimal, first-person voice

If information is not present in the CV, the assistant will explicitly say so.

---

## What this does *not* do

- No web browsing
- No external data sources
- No analytics or tracking
- No storage of conversations
- No personal data collection beyond the local CV file

Everything runs locally except model inference.

---

cv_bot.py contains the AI logic and retrieval pipeline, while app.py provides the Streamlit-based user interface.

## Project structure

```text
.
cv_bot/
├── app.py               # Streamlit UI (browser interface)
├── cv_bot.py            # Core AI logic (LangChain + Gemini)
├── my_cv.txt            # CV content used for retrieval
├── policy_block.txt     # System policy & behavior rules
├── my_cv_database/      # Persistent Chroma vector store
├── .env                 # Environment variables (API keys)
├── check_tools.py         # Check libraries 
├── requirements.txt 
└── README.md



Please check with check_tools.py for libraries.
pip install python-dotenv langchain langchain-core langchain-community langchain-text-splitters langchain-google-genai chromadb
pip installstreamlit
pip install google-api-python-client google-auth