# Asar AI CV Assistant

A minimal, privacy-respecting AI chatbot that answers questions about my professional background using **only my CV as a source of truth**.

Built with:
- Python
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

## Project structure

```text
.
├── main.py                # Chat loop + RAG pipeline
├── my_cv.txt              # Plain-text CV (source of truth)
├── policy_block.txt       # Non-negotiable system rules
├── my_cv_database/        # Local Chroma vector store
├── .env                   # API keys (not committed)
└── README.md
Please check with check_tools.py for libraries.