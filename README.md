---

Asar AI CV Assistant

A minimal, AI chatbot that answers questions about my professional background using only your CV as the source of truth.

This is a personal AI brand assistant, not a generic chatbot.


---

What this is

A CV-aware AI assistant powered by RAG (Retrieval-Augmented Generation)

Answers questions strictly based on a local CV file

Designed to be calm, factual, and minimal

No browsing, no guessing, no hallucination


If a fact is not present in the CV, the assistant explicitly says so.


---

What this does

Answers questions about background, experience, and projects

Uses embeddings + vector search over a CV text file

Enforces strict factual grounding

Maintains short conversational memory (last ~10 messages)

Responds in a first-person, professional tone



---

What this does not do

No web browsing

No external data sources

No analytics or tracking

No user data storage

No conversation logging


Everything runs locally except model inference.


---

This project is meant to be reusable.
To use it with your own information:
Replace the contents of my_cv.txt with your own CV or background text
Delete the my_cv_database/ folder (this clears old embeddings)
Restart the app — embeddings will be rebuilt automatically
No other code changes are required.

---

Tech stack

Python 3.12+

LangChain

Google Gemini (chat + embeddings)

ChromaDB (local vector store)

Streamlit (UI)



---

Project structure

.
├── app.py               # Streamlit UI
├── cv_bot.py            # Core AI logic (RAG pipeline)
├── my_cv.txt            # CV content (replace with your own)
├── policy_block.txt     # System behavior & grounding rules
├── my_cv_database/      # Local Chroma vector store (auto-generated)
├── .env                 # Environment variables (API keys)
├── requirements.txt
└── README.md


---

Local setup (terminal)

1. Create a virtual environment (recommended)



python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

2. Install dependencies



pip install -r requirements.txt

3. Create a .env file



to your .env file: GOOGLE_API_KEY=your_google_api_key

4. Run the CLI version



python cv_bot.py


---

Run with Streamlit (UI)

streamlit run app.py

Then open the browser link shown in the terminal.


---

Replace CV content

To use this project with your own information:

1. Replace my_cv.txt with your own CV content


2. Delete the my_cv_database/ folder


3. Restart the app (embeddings will regenerate automatically)




---

Design principles

Truth over fluency

Minimal UI

No hidden data

No guessing

No lock-in



---

Notes

my_cv_database/ is generated automatically

Do not commit .env to GitHub

This project is intentionally simple and auditable



---
