import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# CORRECTED IMPORT: Reaching into the 'Core' drawer for the string cleaner
from langchain_core.output_parsers import StrOutputParser

# 1. LOAD SECRETS
load_dotenv()
# Grabbing your private info from the .env file
user_phone = os.getenv("USER_PHONE")
user_email = os.getenv("USER_EMAIL")
contact_info = f"Phone: {user_phone}\nEmail: {user_email}"

# 2. PREPARE THE CV DATA
with open("my_cv.txt", "r") as file:
    content = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(content)

# 3. SETUP THE BRAINS (2025 Stable Versions)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# 4. SETUP THE FILING CABINET (Vector Store)
persist_dir = "my_cv_database"

if os.path.exists(persist_dir):
    print("--- Loading existing database from disk... ---")
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    print("--- Creating new database (Asking Google for Embeddings)... ---")
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_dir)

retriever = vectorstore.as_retriever()

# 5. THE CONTEXT COMBINER FUNCTION
def get_full_context(question):
    # This finds relevant snippets from your CV
    docs = retriever.invoke(question)
    cv_text = "\n\n".join(doc.page_content for doc in docs)
    # This adds your private contact info to the snippets
    return f"{cv_text}\n\nPRIVATE CONTACT INFO:\n{contact_info}"

# 6. THE INSTRUCTIONS (Brand Ambassador)
template = """
You are the personal AI Brand Ambassador for Asar Aygul. 
Your goal is to represent Asar in a professional, friendly, and impressive way.

1. Always refer to Asar by name (or use 'he/him'), never call him 'the individual.'
2. Use the provided context to get the facts about his education and experience.
3. If a user asks for an opinion (like 'Is this a good education?'), use your internal 
   knowledge of the job market to provide a positive evaluation of Asar's background.

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 7. THE CONVEYOR BELT (Chain)
chain = (
    {
        "context": get_full_context, 
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# 8. THE INTERACTIVE LOOP
print("\n--- Asar's AI CV is Online! (Type 'quit' to stop) ---")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break
    
    # We pass the input string, and our 'get_full_context' handles the secrets!
    response = chain.invoke(user_input)
    print(f"Bot: {response}")
