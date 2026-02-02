import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from cv_bot import chain

# --- 2026 Minimalist CSS Button Layout ---
st.markdown("""
    <style>
    /* 1. FORCE THE 2-COLUMN GRID (Fixed for Mobile) */
    [data-testid="stHorizontalBlock"] {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important; /* Forces 2 equal halves */
        gap: 6px !important; /* Tightens space between buttons */
        width: 100% !important;
    }

    /* 2. PREVENT STACKING & OVERFLOW */
    [data-testid="column"] {
        width: 100% !important;
        flex: none !important;
    }

    /* 3. SLIM GHOST BUTTONS (Midnight Blue) */
    div.stButton > button {
        width: 100% !important;
        height: 2.2rem !important; /* Keeps height minimalist */
        padding: 0px 2px !important; /* Minimal horizontal padding */
        
        /* FONT SCALING FIX */
        font-size: 9px !important; /* Slightly smaller to fit long words */
        white-space: nowrap !important; /* Prevents text from breaking into 2 lines */
        overflow: hidden !important;
        text-overflow: ellipsis !important; /* Adds '...' if text is way too long */
        
        border-radius: 4px !important;
        background-color: white !important;
        color: #2c3e50 !important; /* Your Midnight Blue */
        border: 1px solid #dfe1e5 !important;
    }

    /* 4. HOVER STATE */
    div.stButton > button:hover {
        border-color: #34495e !important;
        background-color: #f8f9fa !important;
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
# Put all buttons in one 'st.columns(2)' block
cols = st.columns(2)

with cols[0]:
    if st.button("🚢 Royal Caribbean", use_container_width=True):
        st.session_state.pill_selection = "Tell me about your role at Royal Caribbean."
        st.rerun()
    if st.button("🏢 Nestlé", use_container_width=True):
        st.session_state.pill_selection = "Summarize your Nestlé experience."
        st.rerun()
    if st.button("🗂️ CRM / Analytics", use_container_width=True):
        st.session_state.pill_selection = "Describe your CRM experience."
        st.rerun()
    if st.button("🛠️ Top Projects", use_container_width=True):
        st.session_state.pill_selection = "Tell me about your top projects."
        st.rerun()

with cols[1]:
    if st.button("📱 Apps", use_container_width=True):
        st.session_state.pill_selection = "What apps have you worked on?"
        st.rerun()
    if st.button("🧩 Platform", use_container_width=True):
        st.session_state.pill_selection = "Describe your platform trade-off decisions."
        st.rerun()
    if st.button("🧠 Skills", use_container_width=True):
        st.session_state.pill_selection = "What are your strongest skills?"
        st.rerun()
    if st.button("🌍 Global Reach", use_container_width=True):
        st.session_state.pill_selection = "Which markets have you worked in?"
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