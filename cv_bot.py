import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

WORKING_MODEL_NAME = "gemini-2.5-flash-lite"

if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("Missing GOOGLE_API_KEY in environment (.env)")

# 2. PREPARE THE CV DATA
with open("my_cv.txt", "r", encoding="utf-8") as file:
    content = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(content)

#2.1 Prepare Policy
with open("policy_block.txt", "r", encoding="utf-8") as f:
    POLICY_BLOCK = f.read()

# 3. SETUP THE BRAINS
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
model = ChatGoogleGenerativeAI(model=WORKING_MODEL_NAME)

# 4. SETUP THE FILING CABINET (Vector Store)
persist_dir = "my_cv_database"
if os.path.exists(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
else:
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=persist_dir)
    vectorstore.persist()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

def get_full_context(input_data):
    actual_question = input_data["question"]
    docs = retriever.invoke(actual_question)
    cv_text = "\n\n".join(doc.page_content for doc in docs)
    return cv_text

def format_history(history):
    lines = []
    for m in history:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)

template = f"""
=== SYSTEM POLICY (NON-NEGOTIABLE) ===
{POLICY_BLOCK}
=== END POLICY ===

You are the personal AI Brand Ambassador for Asar Aygul.

ROLE RULES:
- Speak in the first person ("I")
- Minimal, calm, professional tone
- No emojis, no hype

STRICT RULES:
1. For ALL FACTS (Schools, Dates, Jobs), you MUST only use the 'Context from CV' provided below.
2. If the 'Context from CV' does not mention a specific fact, say 'Asar hasn't provided that information yet.'
3. You may offer general professional opinions, clearly labeled as opinion.
4. Always speak in the first person ("I") as Asar's representative.
5. Default to 3–6 concise bullet points unless the user explicitly asks for a paragraph.

Chat History:
{{chat_history}}

Context from CV:
{{context}}

Question: {{question}}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": get_full_context,
        "question": lambda x: x["question"],
        "chat_history": lambda x: format_history(x["chat_history"])
    }
    | prompt
    | model
    | StrOutputParser()
)

if __name__ == "__main__":
    chat_history = []
    print("\n--- Asar's AI CV is Online! (Memory Active) ---")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "quit":
            break

        response = chain.invoke({
            "question": user_input,
            "chat_history": chat_history
        })

        print(f"Bot: {response}")



        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]