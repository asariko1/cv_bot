import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from cv_bot import chain

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
    *Current Focus:*
    Asar is currently leading mobile and web platforms at *Royal Caribbean Group*, managing regional operations.
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
st.write("---") # Visual separator
cols = st.columns(3)
pill_selection = None

with cols[0]:
    if st.button("🛠️ Top Projects"):
        pill_selection = "Tell me about your top projects like EchoPath and RedCast."
with cols[1]:
    if st.button("📊 Experience"):
        pill_selection = "What was your role at Royal Caribbean Group?"
with cols[2]:
    if st.button("🌍 Global Reach"):
        pill_selection = "Which international markets have you worked in?"

# If a pill is clicked, treat it like user input
if pill_selection:
    user_input = pill_selection
# ---- END QUICK REPLIES (Pills) ----




# ---- Bottom input ----
# ---- Process Input (Typing OR Pill Click) ----
# This line captures the text input from the user
chat_input = st.chat_input("Type your question…")

# This logic decides which input to use: the typed text or the pill clicked
final_input = chat_input or pill_selection

if final_input:
    # 1. Show user message immediately
    with st.chat_message("user"):
        st.markdown(final_input)

    # 2. Get the response from your LangChain bot
    # We use chat_history before adding the current turn to memory
    history_before = list(st.session_state.chat_history)

    with st.chat_message("assistant"):
        response = chain.invoke({
            "question": final_input,
            "chat_history": history_before
        })
        st.markdown(response)

    # 3. Save both to memory
    st.session_state.chat_history.append(HumanMessage(content=final_input))
    st.session_state.chat_history.append(AIMessage(content=response))

    # 4. Limit history and Rerun to refresh the chat UI
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]
    
    st.rerun()