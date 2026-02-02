import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from cv_bot import chain

# --- CSS Button Lay out ---
st.markdown("""
    <style>
    /* Reduce button height and padding */
    div.stButton > button {
        height: 3em;
        padding-top: 0px;
        padding-bottom: 0px;
        border-radius: 8px;
        border: 1px solid #dfe1e5;
        background-color: white;
        color: #2c3e50; /* Your preferred midnight blue */
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    
    /* Subtle hover effect for modern feel */
    div.stButton > button:hover {
        border-color: #34495e;
        background-color: #f8f9fa;
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- End of CSS design ---



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
    **Current Focus:** Asar is currently leading mobile and web platforms at **Royal Caribbean Group**, managing regional operations.
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

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# ---- QUICK REPLIES (Pills) ----
# 1. INITIALIZE SESSION STATE (Put this near your chat_history initialization)
if "pill_selection" not in st.session_state:
    st.session_state.pill_selection = None

# ---- Pills buttons starts from here ----
st.write("---") 

# 2. CAPTURE & CLEAR: Pull the selection from session state and reset it immediately
pill_selection = st.session_state.pill_selection
st.session_state.pill_selection = None

# 3. UI LAYOUT: 2-Column Button Grid
col1, col2 = st.columns(2)

with col1:
    if st.button("🚢 Royal Caribbean", use_container_width=True):
        st.session_state.pill_selection = "Tell me about your current role at Royal Caribbean and what you own end-to-end."
        st.rerun()
    if st.button("🏢 Nestlé", use_container_width=True):
        st.session_state.pill_selection = "Summarize your Nestlé experience and what outcomes you drove."
        st.rerun()
    if st.button("🗂️ CRM / Analytics", use_container_width=True):
        st.session_state.pill_selection = "Describe your CRM / analytics experience and how you used data to drive decisions."
        st.rerun()
    if st.button("🛠️ Top Projects", use_container_width=True):
        st.session_state.pill_selection = "Tell me about your top projects like EchoPath and RedCast."
        st.rerun()

with col2:
    if st.button("📱 Apps", use_container_width=True):
        st.session_state.pill_selection = "What apps have you worked on (professional + personal)? Summarize briefly."
        st.rerun()
    if st.button("🧩 Platform/Architecture", use_container_width=True):
        st.session_state.pill_selection = "Describe how you make platform decisions and trade-offs with engineering."
        st.rerun()
    if st.button("🧠 Skills", use_container_width=True):
        st.session_state.pill_selection = "What are your strongest skills? Keep it recruiter-friendly in bullets."
        st.rerun()
    if st.button("🌍 Global Reach", use_container_width=True):
        st.session_state.pill_selection = "Which international markets have you worked in?"
        st.rerun()

# ---- PROCESS INPUT (Typing OR Button Click) ----
chat_input = st.chat_input("Type your question...")

# This chooses whichever input is active (the text box or the button clicked)
final_input = chat_input or pill_selection

if final_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(final_input)

    # Get response from the bot
    history_before = list(st.session_state.chat_history)

    with st.chat_message("assistant"):
        # This calls your langchain chain in cv_bot.py
        response = chain.invoke({
            "question": final_input,
            "chat_history": history_before
        })
        st.markdown(response)

    # Save to history
    st.session_state.chat_history.append(HumanMessage(content=final_input))
    st.session_state.chat_history.append(AIMessage(content=response))

    # Keep history manageable (last 20 messages)
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]
    
    # Rerun to update the UI
    st.rerun()