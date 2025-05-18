
import streamlit as st
import os
import requests
import json
import time

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Config ---
API_KEY = st.secrets["api_key"]
API_ENDPOINT = st.secrets["api_endpoint"]  # e.g. https://<your-resource>.openai.azure.com
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
            if file.endswith((".txt", ".html", ".xml")):
                path = os.path.join(directory, file)
                loader = TextLoader(path)
                documents.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(documents)

    desktop_chunks = load_and_chunk("docs/one-desktop-plans")
    webapp_chunks = load_and_chunk("docs/webapp-transformation-plans")

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    desktop_db = FAISS.from_documents(desktop_chunks, embeddings)
    webapp_db = FAISS.from_documents(webapp_chunks, embeddings)

    return desktop_db.as_retriever(), webapp_db.as_retriever()

desktop_retriever, webapp_retriever = load_vectorstores()

# --- OpenAI Completion ---
def call_openai(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are Ata-cat, a friendly and expert assistant in Ataccama ONE Desktop and WebApp transformation plans."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 800
    }
    url = f"{API_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        return f"[ERROR {response.status_code}]: {response.text}"
    return response.json()["choices"][0]["message"]["content"]

# --- UI Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = []

st.write("Hi! I'm :violet[Ata-cat]. What can I help you build today?")

plan_type = st.selectbox("Select your transformation type:", ["desktop", "webapp"])
user_input = st.chat_input("Type your question...")

if user_input:
    st.markdown(f"**You asked:** {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"🧭 **Selected plan type:** `{plan_type}`")

    # Step 1: Retrieve Relevant Docs
    with st.spinner("Retrieving relevant information..."):
        if plan_type == "desktop":
            docs = desktop_retriever.get_relevant_documents(user_input)
        elif plan_type == "webapp":
            docs = webapp_retriever.get_relevant_documents(user_input)
        else:
            docs = []
        context_text = "\n\n".join([doc.page_content for doc in docs[:3]])

    # Step 2: Generate Final Answer
    with st.spinner("Writing a step-by-step answer..."):
        solution_prompt = (
            f"The user asked: {user_input}\n\n"
            f"Relevant context from documentation:\n{context_text}\n\n"
            f"Please explain the answer clearly, step-by-step, in a non-technical tone. "
            f"Include examples, best practices, and useful Ataccama components where appropriate."
        )
        response = call_openai(solution_prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)

# --- Chat History ---
with st.expander("Conversation history"):
    for msg in st.session_state.messages:
        st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")
