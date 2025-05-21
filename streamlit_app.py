
import streamlit as st
import os
import requests
import json
import hashlib
import faiss

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, CSVLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI

# --- Config ---
API_KEY = st.secrets["api_key"]
API_ENDPOINT = st.secrets["api_endpoint"]
DEPLOYMENT_NAME = "gpt-4.1-mini"
API_VERSION = "2024-02-15-preview"

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Ata-transform", page_icon="images/ataccama_logo.png")
st.title("Welcome to :violet[Ataccama's] Transformation Assistant Bot")

# --- Load Vector Stores ---
@st.cache_resource
def load_vectorstores():
    def try_fallback_pdf_loader(path):
        try:
            return PyMuPDFLoader(path)
        except Exception:
            print(f"🔁 Falling back to OCR for: {path}")
            return UnstructuredPDFLoader(path, strategy="ocr_only")

    def load_and_chunk(directory, index_name):
        index_path = f"{index_name}.faiss"

        if os.path.exists(index_path) and os.path.exists(f"{index_name}.pkl"):
            return FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)

        documents = []
        loader_map = {
            ".txt": lambda path: TextLoader(path, encoding="utf-8"),
            ".html": lambda path: TextLoader(path, encoding="utf-8"),
            ".xml": lambda path: TextLoader(path, encoding="utf-8"),
            ".json": lambda path: TextLoader(path, encoding="utf-8"),
            ".properties": lambda path: TextLoader(path, encoding="utf-8"),
            ".pdf": lambda path: try_fallback_pdf_loader(path),
            ".csv": lambda path: CSVLoader(file_path=path),
        }

        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            ext = os.path.splitext(file)[1].lower()
            loader_fn = loader_map.get(ext)
            if loader_fn:
                try:
                    loader = loader_fn(path)
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata["source"] = file
                    documents.extend(loaded_docs)
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
            else:
                print(f"⚠️ Skipped unsupported file type: {file}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(index_name)
        return db

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    desktop_db = load_and_chunk("docs/one-desktop-plans", "desktop_index")
    webapp_db = load_and_chunk("docs/webapp-transformation-plans", "webapp_index")

    return desktop_db, webapp_db

desktop_db, webapp_db = load_vectorstores()

# --- Interpreter Agent (Scoped to Ataccama Steps) ---
@st.cache_data(show_spinner=False)
def interpret_question(original_query: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    system_prompt = (
        "You are an expert query rewriting agent for an Ataccama-specific assistant. "
        "Your task is to take the user's original search query and rewrite it to be more effective for retrieving relevant information "
        "from an Ataccama transformation plan knowledge base.\n\n"
        "Stay strictly within the scope of Ataccama ONE Desktop and WebApp capabilities.\n\n"
        f"Original query:\n\"{original_query}\"\n\n"
        "Rewrite this query to best retrieve Ataccama-specific documentation. Return only the rewritten query."
    )

    payload = {
        "messages": [{"role": "system", "content": system_prompt}],
        "temperature": 0.3,
        "max_tokens": 100
    }

    url = f"{API_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return original_query
    return response.json()["choices"][0]["message"]["content"].strip()

# --- Build a ReAct Agent with Document Tool ---
def create_react_agent(retriever_db, plan_type: str, tone: str):
    def search_docs(query: str) -> str:
        docs = retriever_db.similarity_search(query, k=4)
        return "\n\n".join([doc.page_content for doc in docs])

    tools = [
        Tool(
            name="SearchDocs",
            func=search_docs,
            description=f"Search the Ataccama {plan_type} documentation to find info about transformation steps, components, or behavior."
        )
    ]

    llm = AzureChatOpenAI(
        azure_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        deployment_name=DEPLOYMENT_NAME,
        openai_api_version=API_VERSION,
        temperature=0.4,
        model_kwargs={"top_p": 0.9}
    )

    system_msg = (
        "You are Ata-cat, a smart and friendly Ataccama assistant for transformation plan questions.\n"
        "Use the `SearchDocs` tool to find relevant info from Ataccama docs.\n"
        "Think step-by-step. If needed, make multiple searches to answer fully.\n\n"
        "Always use the exact Ataccama transformation step names, like 'Alter Format', 'Representative Creator', 'Column Assigner', etc.\n"
        f"Respond in this tone: {tone}.\n"
    )

    agent = initialize_agent(
        tools,
        llm,
        agent="chat-zero-shot-react-description",
        verbose=False,
        handle_parsing_errors=True,
        agent_kwargs={"system_message": system_msg}
    )

    return agent

# --- UI Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("Hi! I'm :violet[Ata-cat]. What can I help you build today?")

plan_type = st.selectbox("Select your transformation type:", ["desktop", "webapp"])

tone = st.select_slider(
    "🗣️ Choose response tone:",
    options=["Very Concise", "Concise", "Neutral", "Chatty", "Very Chatty"],
    value="Neutral"
)

tone_map = {
    "Very Chatty": "Be extremely detailed, step-by-step with examples and analogies.",
    "Chatty": "Be clear and friendly with component names.",
    "Neutral": "Balance detail and brevity with Ataccama-specific clarity.",
    "Concise": "Be brief and stick to component names.",
    "Very Concise": "Just list the required Ataccama steps plainly."
}
tone_instruction = tone_map[tone]

user_input = st.chat_input("Type your question...")

if user_input:
    with st.spinner("Interpreting your question..."):
        interpreted_question = interpret_question(user_input)

    st.markdown(f"**You asked:** {user_input}")

    with st.spinner("Reasoning and searching with Ata-cat agent..."):
        retriever_db = desktop_db if plan_type == "desktop" else webapp_db
        agent = create_react_agent(retriever_db, plan_type, tone_instruction)
        try:
            response = agent.run(interpreted_question)
        except Exception as e:
            response = f"🤖 Sorry, there was an error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)

with st.expander("Conversation history"):
    for msg in st.session_state.messages:
        st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")
