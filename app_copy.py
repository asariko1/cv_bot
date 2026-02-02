import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from cv_bot import chain
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Asar's CV Bot", page_icon="🚀")

# ---- SIDEBAR: Contact & Location Info ----
with st.sidebar:
    st.image("https://github.com/asariko1.png", width=100) # Your GitHub Profile Pic
    st.title("Aşar Aygül")
    st.subheader("Digital Manager | Royal Caribbean")
    
    st.markdown("""
    📍 **Locations:** Istanbul 🇹🇷 / Miami 🇺🇸  
    📧 **Email:** asaraygul@gmail.com  
    🔗 [LinkedIn Profile](https://www.linkedin.com/in/asar-aygul)  
    🌐 [asariko.net](https://asariko.net)
    
    ---
    **Global Market Experience:** US, Mediterranean, North Europe, Australia/NZ.
    **Languages:** Turkish - English
    """)

    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ----END of SIDEBAR: Contact & Location Info ----

st.title("Asar’s AI CV Bot")
st.caption("Ask about my background, projects, or skills. Answers are grounded only in my CV.")

# -----------------------------
# Quick prompts (pill-like buttons)
# -----------------------------
st.markdown("#### Quick prompts")
q1, q2, q3 = st.columns(3)

def queue(text: str):
    st.session_state.queued_input = text

with q1:
    if st.button("🚢 Royal Caribbean", use_container_width=True):
        queue("Tell me about your current role at Royal Caribbean and what you own end-to-end.")
    if st.button("📱 Apps", use_container_width=True):
        queue("What apps have you worked on (professional + personal)? Summarize briefly.")

with q2:
    if st.button("🏢 Nestlé", use_container_width=True):
        queue("Summarize your Nestlé experience and what outcomes you drove.")
    if st.button("🧩 Platform/Architecture", use_container_width=True):
        queue("Describe how you make platform decisions and trade-offs with engineering.")

with q3:
    if st.button("🗂️ CRM / Analytics", use_container_width=True):
        queue("Describe your CRM / analytics experience and how you used data to drive decisions.")
    if st.button("🧠 Skills", use_container_width=True):
        queue("What are your strongest skills? Keep it recruiter-friendly in bullets.")

st.markdown("---")

# -----------------------------
# END of Quick prompts (pill-like buttons)
# -----------------------------
# After buttons:

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# ---- For bUttons State ----
if "queued_input" not in st.session_state:
    st.session_state.queued_input = None


#  Auto-greeting on first load (only once)
if len(st.session_state.chat_history) == 0:
    st.session_state.chat_history.append(
        AIMessage(content="Hello there! 👋 I'm Asar’s AI CV assistant. I can help you learn about my background, experience, or projects. What would you like to know?")
    )


# ---- Clear button (simple, standard) ----
if st.button("Clear chat"):
    st.session_state.chat_history = []
    st.rerun()

# ---- Render history (this is the part you were missing) ----
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---- Bottom input ----
user_input = st.chat_input("Type your question…")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # IMPORTANT: use history BEFORE adding this user input
    history_before = list(st.session_state.chat_history)

    response = chain.invoke({
        "question": user_input,
        "chat_history": history_before
    })

    with st.chat_message("assistant"):
        st.markdown(response)

    # Now save both to memory (in correct order)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

    # Keep last 20 messages (10 turns)
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]
