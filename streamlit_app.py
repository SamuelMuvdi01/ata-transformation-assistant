
import streamlit as st
import os
import requests
import json
import re
import hashlib

from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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
    def load_and_chunk(directory):
        documents = []
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            try:
                if file.endswith((".txt", ".html", ".xml", ".json")):
                    loader = TextLoader(path, encoding="utf-8")
                    documents.extend(loader.load())
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    documents.extend(loader.load())
                elif file.endswith(".csv"):
                    loader = CSVLoader(file_path=path)
                    documents.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(documents)

    desktop_chunks = load_and_chunk("docs/one-desktop-plans")
    webapp_chunks = load_and_chunk("docs/webapp-transformation-plans")

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    desktop_db = FAISS.from_documents(desktop_chunks, embeddings)
    webapp_db = FAISS.from_documents(webapp_chunks, embeddings)

    return desktop_db.as_retriever(), webapp_db.as_retriever()

desktop_retriever, webapp_retriever = load_vectorstores()

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
        "Your goal is to help identify the most relevant Ataccama transformation step or component. "
        "Stay strictly within the scope of Ataccama ONE Desktop and WebApp capabilities.\n\n"
        "Avoid mentioning or implying non-Ataccama tools (e.g., Tableau, Excel, Power BI).\n\n"
        f"Here is the user's original query:\n\"{original_query}\"\n\n"
        "Rewrite this query to best retrieve Ataccama-specific documentation. Respond with only the rewritten query."
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

# --- Final Answer Generator with Step-Specific Prompt ---
@st.cache_data(show_spinner=False)
def call_openai_solution(prompt: str, cache_key: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Ata-cat, a friendly and expert assistant in Ataccama ONE Desktop and WebApp transformation plans.\n"
                    "Always name the exact transformation step(s) when applicable — like 'Alter Format', 'Filter', 'Column Assigner', 'Deduplicate'. "
                    "Avoid vague terms like 'Transformation Step' or 'Clean Step'. Use Ataccama terms only.\n\n"
                    "If helpful, explain the steps in clear, numbered lists. Stick strictly to Ataccama features."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 1000
    }
    url = f"{API_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return f"[ERROR {response.status_code}]: {response.text}"
    return response.json()["choices"][0]["message"]["content"]

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

tone_instruction_map = {
    "Very Chatty": "Be extremely detailed, explain the plan step-by-step, use examples, and describe the function of each Ataccama component.",
    "Chatty": "Be friendly and clear, include best practices and component names.",
    "Neutral": "Balance detail and brevity. Explain clearly and provide examples.",
    "Concise": "Be brief and focus only on core transformation components.",
    "Very Concise": "List only the minimum transformation steps in plain text."
}
tone_instruction = tone_instruction_map[tone]

user_input = st.chat_input("Type your question...")

if user_input:
    with st.spinner("Interpreting your question..."):
        interpreted_question = interpret_question(user_input)

    st.markdown(f"**You asked:** {user_input}")

    with st.spinner("Retrieving relevant documentation..."):
        retriever = desktop_retriever if plan_type == "desktop" else webapp_retriever
        docs = retriever.get_relevant_documents(interpreted_question)
        context_text = "\n\n".join([doc.page_content for doc in docs[:3]])

    with st.spinner("Writing a step-by-step answer..."):
        examples = (
            "Examples:\n"
            "- To add or remove columns → use 'Alter Format'\n"
            "- To rename a column → use 'Column Assigner'\n"
            "- To filter records → use 'Filter'\n"
            "- To remove duplicates → use 'Deduplicate'\n\n"
        )
        full_prompt = (
            f"{examples}"
            f"The user originally asked: {user_input}\n"
            f"Interpreted and cleaned question: {interpreted_question}\n\n"
            f"Relevant context from documentation:\n{context_text}\n\n"
            f"{tone_instruction}\n\n"
            f"Be accurate. Mention exact Ataccama transformation steps if relevant. "
            f"Do not suggest or invent technologies outside the Ataccama platform."
        )

        cache_key = hashlib.sha256((interpreted_question + tone).encode()).hexdigest()
        response = call_openai_solution(full_prompt, cache_key)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)

with st.expander("Conversation history"):
    for msg in st.session_state.messages:
        st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")
