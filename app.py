import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# 1) Load Streamlit secrets into env BEFORE importing cv_bot
# (important: cv_bot may read env at import time)
for k in ["GOOGLE_API_KEY"]:
    if k in st.secrets:
        os.environ[k] = str(st.secrets[k])

from cv_bot import chain  # import after env is set

st.set_page_config(page_title="Asar's CV Bot", page_icon="💬", layout="centered")

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Asar's AI CV is Online!")

# ---- Render history ----
for m in st.session_state.chat_history:
    if isinstance(m, HumanMessage):
        with st.chat_message("user"):
            st.markdown(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(m.content)

# ---- Bottom input ----
user_input = st.chat_input("Ask about my background, projects, skills…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    response = chain.invoke({
        "question": user_input,
        "chat_history": st.session_state.chat_history
    })

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]