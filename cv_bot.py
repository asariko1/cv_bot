import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 1. Load secrets
load_dotenv()

# 2. Read the CV
with open("my_cv.txt", "r") as file:
    content = file.read()

# 3. Prepare the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(content)

# 4. Setup the Brains
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# 4.1 PLACE THE DIAGNOSTIC LINE HERE:
print(f"DEBUG: Python is currently using: {embeddings.model}")

# 5. SETUP THE STORE (With a safety check)
persist_dir = "my_cv_database"

# We check if the folder already exists
if os.path.exists(persist_dir):
    print("--- Loading existing database from disk... ---")
    vectorstore = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embeddings
    )
else:
    print("--- Creating new database (Asking Google for Embeddings)... ---")
    vectorstore = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        persist_directory=persist_dir
    )

retriever = vectorstore.as_retriever()

# 6. The "Modern" Prompt
template = """
You are the personal AI Brand Ambassador for Asar Aygul. 
Your goal is to represent Asar in a professional, friendly, and impressive way.

1. Always refer to Asar by name (or use 'he/him'), never call him 'the individual.'
2. Use the provided context to get the facts about his education and experience.
3. If a user asks for an opinion (like 'Is this a good education?'), use your internal 
   knowledge of the job market and university standards to provide a positive 
   and professional evaluation of Asar's background.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 7. THE MODERN CONVEYOR BELT (LCEL)
# This replaces all the 'create_chain' functions!
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)



# 8. The Chat Loop
print("--- Modern AI CV Bot is Online! ---")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
    
    # We invoke our conveyor belt
    response = chain.invoke(user_input)
    print(f"Bot: {response}")

