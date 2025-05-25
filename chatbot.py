import os
import string
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

MODEL = "gpt-4o-mini"
db_name = "vector_db"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load and process PDF
pdf_path = "restaurant_details.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
for doc in documents:
    doc.metadata["doc_type"] = "restaurant_info"
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Vector store
embeddings = OpenAIEmbeddings()
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# Conversation chain
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

def ask_bot(question):
    # Get previous messages from memory
    history = memory.load_memory_variables({}).get("chat_history", [])
    history_dialogue = ""
    for msg in history:
        role = "User" if msg.type == "human" else "Assistant"
        history_dialogue += f"{role}: {msg.content}\n"

    # Greeting/goodbye/thanks detection (only if message is just that)
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    goodbyes = ["bye", "goodbye", "see you", "farewell"]
    thanks = ["thank you", "thanks", "thx", "appreciate"]

    q_lower = question.lower().strip()
    q_clean = q_lower.translate(str.maketrans('', '', string.punctuation))

    if any(q_clean == greet for greet in greetings):
        return "Hello! 👋 How can I assist you with our restaurant information today?"
    if any(q_clean == bye for bye in goodbyes):
        return "Goodbye! 👋 If you have more questions about our restaurant, feel free to ask anytime."
    if any(q_clean == thank for thank in thanks):
        return "You're welcome! 😊 If you need anything else about our restaurant, just let me know."

    # DEBUG: See what the retriever returns
    docs = retriever.get_relevant_documents(question)
    print("Retrieved docs for question:", question)
    for d in docs:
        print(d.page_content[:200])

    prompt = (
        "You are a helpful, friendly, and knowledgeable restaurant assistant. "
        "Answer the user's question using the information from the restaurant_details.pdf below. "
        "If the answer is not found, say 'I'm sorry, I don't have that information.'\n"
        "Think step by step before answering.\n"
        "Use the previous conversation for context and continuity.\n\n"
        f"{history_dialogue}User: {question}"
    )
    result = conversation_chain.invoke({"question": prompt})
    return result["answer"]