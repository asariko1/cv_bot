import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from cv_bot import chain
import os
from dotenv import load_dotenv
from gdrive_log import append_missing_question
from cv_bot import chain, MISSING_PHRASE

load_dotenv()
SHEET_ID = os.getenv("SHEET_ID")
if not SHEET_ID:
    raise RuntimeError("Missing SHEET_ID in environment (.env)")


st.set_page_config(page_title="Asar's CV Bot", page_icon="💬", layout="centered")

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Asar's AI CV is Online!")

# ---- Render history (with icons) ----
for m in st.session_state.chat_history:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.markdown(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(m.content)

# ---- Bottom input (pinned) ----
user_input = st.chat_input("Ask about my background, projects, skills…")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)  # if you want emojy: st.markdown ("🧑‍💻 " + user_input)

    # Run chain
    response = chain.invoke({
        "question": user_input,
        "chat_history": st.session_state.chat_history
    })

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)  # if you want emojy: st.markdown ("🧑🤖 " + user_input)


    # Missing Phrase
    if MISSING_PHRASE in response:
        append_missing_question(SHEET_ID, user_input.strip())
        st.info("Logged unanswered question.")



    # Save to memory
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

    # Keep last 10 messages (same rule)
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]