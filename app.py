import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from cv_bot import chain  # only import the chain; keep logic in cv_bot.py

st.set_page_config(
    page_title="Asar's CV Bot",
    page_icon="🚀",
    layout="centered",
)

# ---- Minimal CSS polish ----
st.markdown(
    """
    <style>
      .block-container { 
        max-width: 820px; 
        padding-top: 2.5rem; 
        padding-bottom: 2rem; 
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Header ----
top_left, top_right = st.columns([1, 0.22], vertical_alignment="center")

with top_left:
    st.title("Asar’s AI CV Bot")
    st.caption("Ask about my background, projects, or skills. Answers are grounded only in my CV.")

with top_right:
    if st.button("Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---- Bottom input (pinned by Streamlit) ----
user_input = st.chat_input("Type your question…")

if user_input:
    # Show user message instantly
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

# Run chain
    response = chain.invoke({
        "question": user_input,
        "chat_history": st.session_state.chat_history
    })

    # Show assistant response
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown(response)

    # Save to memory
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

    # Keep last 10 messages
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]