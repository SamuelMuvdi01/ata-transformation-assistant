import streamlit as st
import os
import requests
import json

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Ata-transform", page_icon="images/ataccama_logo.png")
st.title("Welcome to :violet[Ataccama's] Transformation Assistant Bot")

API_KEY = st.secrets["cipher"]
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# --- Vector Store Loading ---
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

# --- Gemini Query ---
def query_gemini(prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 600
        }
    }

    res = requests.post(GEMINI_URL, headers=headers, json=payload)
    if res.status_code != 200:
        return f"Error {res.status_code}: {res.json()}"
    
    return res.json()["candidates"][0]["content"]["parts"][0]["text"]

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are **Ata-cat**, an expert assistant in Ataccama ONE data transformation logic. "
                "You help users understand and build ONE Desktop plans and WebApp transformation plans. "
                "Always refer to yourself as Ata-cat. Speak clearly, helpfully, and conversationally like a trainer. "
                "For 'how do I' or 'how to' questions, explain in detail:\n"
                "- What the step/component is and what it does\n"
                "- Step-by-step usage instructions\n"
                "- Practical examples\n"
                "- Common use cases and best practices\n"
                "- Related steps or components the user might want to consider (e.g., Filter, Condition, Extract Filter) "
                "and how they compare or integrate in data flows.\n"
                "Make sure to cover both Ataccama ONE Desktop and WebApp perspectives as needed."
            )
        }
    ]

if "plan_type" not in st.session_state:
    st.session_state.plan_type = None

if "pending_request" not in st.session_state:
    st.session_state.pending_request = None

# --- Load Vector DBs ---
desktop_retriever, webapp_retriever = load_vectorstores()

# --- Utility: Plan Detection ---
def detect_plan_type(text):
    text = text.lower()
    if "desktop" in text or "one desktop" in text:
        return "desktop"
    elif "webapp" in text or "transformation plan" in text or "web app" in text:
        return "webapp"
    return None

# --- Utility: Contextual Retrieval ---
def get_contextual_docs(user_input):
    detected = detect_plan_type(user_input)
    if detected:
        st.session_state.plan_type = detected

    if st.session_state.plan_type == "desktop":
        return desktop_retriever.get_relevant_documents(user_input), "desktop"
    elif st.session_state.plan_type == "webapp":
        return webapp_retriever.get_relevant_documents(user_input), "webapp"
    else:
        return [], "ambiguous"

# --- UI + Chat Flow ---
st.write("Hi! I'm :violet[Ata-cat]. What can I help you build today?")
user_request = st.chat_input("Type your question or say hello...")

if user_request:
    context_docs, context_type = get_contextual_docs(user_request)

    if context_type == "ambiguous" and st.session_state.plan_type is None:
        st.session_state.pending_request = user_request
        clarification_prompt = (
            f"The user asked: \"{user_request}\".\n\n"
            "They didn't say whether it's for an Ataccama ONE Desktop plan or a WebApp transformation plan. "
            "Please ask the user in a polite and friendly way to clarify which plan they're asking about."
        )
        response = query_gemini(clarification_prompt)
        st.markdown(f"**You asked:** {user_request}")
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        actual_request = user_request if st.session_state.pending_request is None else st.session_state.pending_request
        st.session_state.pending_request = None

        st.session_state.messages.append({"role": "user", "content": actual_request})
        st.markdown(f"**You asked:** {actual_request}")

        context_text = "\n\n".join([doc.page_content for doc in context_docs[:3]])
        plan = st.session_state.plan_type.upper()

        enriched_prompt = (
            f"You are Ata-cat, an expert assistant helping with Ataccama ONE {plan} transformation plans.\n\n"
            f"The user asked:\n\"{actual_request}\"\n\n"
            f"Relevant documentation:\n{context_text}\n\n"
            f"Answer as a clear and friendly trainer. For 'how to' or 'how do I' questions, please explain in detail:\n"
            f"- What the step or component is and its purpose\n"
            f"- Step-by-step usage instructions\n"
            f"- Practical examples\n"
            f"- Typical use cases and best practices\n"
            f"- Related or alternative steps/components the user might want to consider (for example, Filter, Condition, Extract Filter), "
            f"and explain how they compare or work together in data flows.\n\n"
            f"Make sure to provide information relevant to both Ataccama ONE Desktop and WebApp plans if applicable. "
            f"Speak in a helpful, friendly, and conversational tone."
        )

        with st.spinner("Let me check the docs for you..."):
            response = query_gemini(enriched_prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

# --- Conversation History ---
with st.expander("Conversation history"):
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")