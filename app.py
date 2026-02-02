import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from cv_bot import chain

# --- 2026 Minimalist CSS Button Layout ---
st.markdown("""
    <style>
    /* 1. FORCE THE GRID: Stop vertical stacking on mobile */
    [data-testid="column"] {
        width: calc(50% - 1rem) !important;
        flex: 1 1 calc(50% - 1rem) !important;
        min-width: calc(50% - 1rem) !important;
    }

    /* 2. SLIM BUTTONS: Remove the bulk */
    div.stButton > button {
        border-radius: 4px !important;
        height: 2.2rem !important; /* Extremely slim height */
        line-height: 1 !important;
        padding: 0px !important;
        font-size: 12px !important;
        background-color: #ffffff !important;
        color: #2c3e50 !important; /* Midnight Blue */
        border: 1px solid #e0e0e0 !important;
        box-shadow: none !important;
    }

    /* 3. TIGHTEN SPACING: Pull everything up */
    [data-testid="stHorizontalBlock"] {
        gap: 0.4rem !important;
        margin-bottom: -0.8rem !important;
    }
    
    /* Remove default Streamlit padding that causes extra white space */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
# --- End of Updated CSS design ---


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


# ---- Clear button (simple, standard) ---- (We placed to sidebar, no need this one.)
#if st.button("Clear chat"):
 #   st.session_state.chat_history = []
  #  st.rerun()

# ---- Render history (this is the part you were missing) ----
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# ---- QUICK REPLIES (Modern 2x4 Grid) ----
st.write("---") 

# 1. Initialize session state if not already done (Safety check)
if "pill_selection" not in st.session_state:
    st.session_state.pill_selection = None

# 2. CAPTURE & CLEAR: Pull selection to local variable and reset state
pill_selection = st.session_state.pill_selection
st.session_state.pill_selection = None

# 3. UI LAYOUT: 2-Column Grid (CSS from Step 1 will fix mobile width)
col1, col2 = st.columns(2)

with col1:
    if st.button("🚢 Royal Caribbean 🚢"):
        st.session_state.pill_selection = "Tell me about your current role at Royal Caribbean and what you own end-to-end."
        st.rerun()
    if st.button("🏢 Nestlé 🏢"):
        st.session_state.pill_selection = "Summarize your Nestlé experience and what outcomes you drove."
        st.rerun()
    if st.button("🗂️ CRM / Analytics 🗂️"):
        st.session_state.pill_selection = "Describe your CRM / analytics experience and how you used data to drive decisions."
        st.rerun()
    if st.button("🛠️ Projects 🛠️"):
        st.session_state.pill_selection = "Tell me about your top projects like Chatbot, EchoPath and RedCast."
        st.rerun()

with col2:
    if st.button("📱 Apps 📱"):
        st.session_state.pill_selection = "What apps have you worked on (professional + personal)? Summarize briefly."
        st.rerun()
    if st.button("🧩 Turkcell 🧩"):
        st.session_state.pill_selection = "Summarize your Tukcell experience and what outcomes you drove."
        st.rerun()
    if st.button("🧠 Skills 🧠"):
        st.session_state.pill_selection = "What are your strongest skills? Keep it recruiter-friendly in bullets."
        st.rerun()
    if st.button("🌍 Global Reach   "):
        st.session_state.pill_selection = "Which international markets have you worked in?"
        st.rerun()
# ---- END QUICK REPLIES ----


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