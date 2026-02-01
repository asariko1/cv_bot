import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from cv_bot import chain  # only import the chain; keep logic in cv_bot.py

st.set_page_config(
    page_title="Asariko CV Bot",
    page_icon="🧑‍💻",
    layout="centered",
)

# ---- Minimal CSS polish ----
st.markdown(
    """
    <style>
      /* Narrower content, more premium */
      .block-container { max-width: 820px; padding-top: 2.0rem; padding-bottom: 2.0rem; }

      /* Slightly tighten overall spacing */
      [data-testid="stVerticalBlock"] { gap: 0.75rem; }

      /* Make the title area feel less “Streamlit demo” */
      .asar-title { font-size: 1.6rem; font-weight: 650; margin: 0; line-height: 1.2; }
      .asar-sub { color: rgba(0,0,0,0.55); margin-top: 0.25rem; margin-bottom: 0.75rem; }

      /* Optional: remove extra top padding Streamlit sometimes adds */
      header { visibility: hidden; height: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Header ----
col1, col2 = st.columns([1, 0.22], vertical_alignment="center")
with col1:
    st.markdown('<p class="asar-title">Asariko CV Bot</p>', unsafe_allow_html=True)
    st.markdown('<div class="asar-sub">Ask about my background, projects, skills. I answer using only my CV.</div>', unsafe_allow_html=True)

with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---- Render history with avatars ----
for m in st.session_state.chat_history:
    if isinstance(m, HumanMessage):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(m.content)
    elif isinstance(m, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(m.content)

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
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response)

    # Save to memory
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

    # Keep last 10 messages
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]