import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# NEW: Imports for labeling the conversation history
from langchain_core.messages import HumanMessage, AIMessage

# Define our preferred working model here (Using the latest one you confirmed)
WORKING_MODEL_NAME = "gemini-flash-lite-latest"  # <--- USE THIS NAME BASED ON YOUR SUCCESS

# 1. LOAD SECRETS
load_dotenv()
user_phone = os.getenv("USER_PHONE")
user_email = os.getenv("USER_EMAIL")
contact_info = f"Phone: {user_phone}\nEmail: {user_email}"

# 2. PREPARE THE CV DATA
with open("my_cv.txt", "r") as file:
    content = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(content)

# 3. SETUP THE BRAINS
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
model = ChatGoogleGenerativeAI(model=WORKING_MODEL_NAME)

# 4. SETUP THE FILING CABINET (Vector Store)
persist_dir = "my_cv_database"
if os.path.exists(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_dir)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 5. THE CONTEXT COMBINER (Now handles the 'box' input)
def get_full_context(input_data):
    # We "open the box" to get the question string
    actual_question = input_data["question"]
    docs = retriever.invoke(actual_question)
    cv_text = "\n\n".join(doc.page_content for doc in docs)
    return f"{cv_text}\n\nPRIVATE CONTACT INFO:\n{contact_info}"

# 6. THE INSTRUCTIONS (Updated to include a slot for History)
template = """
You are the personal AI Brand Ambassador for Asar Aygul. 

STRICT RULES:
1. For ALL FACTS (Schools, Dates, Jobs), you MUST only use the 'Context from CV' provided below. 
2. If the 'Context from CV' does not mention a specific fact, say 'Asar hasn't provided that information yet.' 
3. You may use your knowledge by searching to provide professional opinions (e.g., 'Is this a good school?').
4. Always speak in the first person ("I") as Asar's representative.

Chat History:
{chat_history}

Context from CV:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 7. THE CONVEYOR BELT (Chain)
# We use 'lambda' to tell the chain which part of the input goes where
chain = (
    {
        "context": get_full_context, 
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | model
    | StrOutputParser()
)

# 8. THE INTERACTIVE LOOP
chat_history = [] # This is our "Memory Box"
print("\n--- Asar's AI CV is Online! (Memory Active) ---")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
    
    # We pass BOTH the question and the current history
    response = chain.invoke({
        "question": user_input, 
        "chat_history": chat_history
    })
    
    print(f"Bot: {response}")

    # SAVE TO MEMORY: We add the latest exchange to the box
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    # Keep only the last 10 messages so the "package" doesn't get too big
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]