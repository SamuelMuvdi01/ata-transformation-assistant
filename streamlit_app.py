import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import requests as req
import pandas as pd
import os

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Streamlit setup
st.set_page_config(page_title="Ata-transform-bot")
st.title("Welcome to Ataccama's Transformation Assistant Bot")

# -------- Cached data loader and vector store builder --------
@st.cache_resource
def load_vectorstores():
    """Load documents and build vector store for both WebApp and Desktop plans."""
    def load_and_chunk(directory):
        documents = []
        for file in os.listdir(directory):
            if file.endswith(".txt") or file.endswith(".html") or file.endswith(".xml"):
                path = os.path.join(directory, file)
                loader = TextLoader(path)
                documents.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(documents)
    
    # Load Desktop and WebApp plan documents
    desktop_chunks = load_and_chunk("docs/one-desktop-plans")
    webapp_chunks = load_and_chunk("docs/webapp-transformation-plans")
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Create and save vector stores for each plan type
    desktop_db = FAISS.from_documents(desktop_chunks, embeddings)
    webapp_db = FAISS.from_documents(webapp_chunks, embeddings)

    # Save local vector index for each plan type
    desktop_db.save_local("vector_index_chunks_desktop")
    webapp_db.save_local("vector_index_chunks_webapp")

    return desktop_db.as_retriever(), webapp_db.as_retriever()

# -------- OpenRouter Integration --------
@st.cache_resource
def get_openrouter_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=st.secrets['deepseek_key']
    )

def query_openrouter(messages):
    client = get_openrouter_client()

    completion = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct:free",
        messages=messages,
        extra_headers={
            "HTTP-Referer": "https://ata-transformation-assistant.streamlit.app/",
            "X-Title": "Ata-transform-bot",
        },
    )

    return completion.choices[0].message.content

# -------- UI Logic --------

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are an expert in Ataccama data transformation logic. Respond clearly and helpfully."
        }
    ]

if "plan_type" not in st.session_state:
    st.session_state.plan_type = None

st.write("Let's start building! What can we help you build today?")
user_request = st.chat_input("Please enter a transformation plan request")

# Load retrievers
desktop_retriever, webapp_retriever = load_vectorstores()

# Plan detection and persistence
def detect_plan_type(user_input):
    input_lower = user_input.lower()
    if "desktop" in input_lower or "one desktop" in input_lower:
        return "desktop"
    elif "webapp" in input_lower or "transformation plan" in input_lower:
        return "webapp"
    return None

def get_contextual_docs(user_input):
    detected = detect_plan_type(user_input)

    # If user clearly specified a plan type, update the session
    if detected:
        st.session_state.plan_type = detected

    # Use session-stored plan type to retrieve docs
    if st.session_state.plan_type == "desktop":
        return desktop_retriever.get_relevant_documents(user_input), "desktop"
    elif st.session_state.plan_type == "webapp":
        return webapp_retriever.get_relevant_documents(user_input), "webapp"
    else:
        return [], "ambiguous"

# Main interaction loop
if user_request:
    try:
        context_docs, context_type = get_contextual_docs(user_request)

        if st.session_state.plan_type is None:
            clarification_msg = (
                "Are you referring to a **ONE Desktop plan** or a **WebApp transformation plan**? "
                "Please clarify so I can retrieve the correct documentation context."
            )
            st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
            st.write(clarification_msg)
        else:
            context = "\n\n".join([doc.page_content for doc in context_docs[:2]])
            enriched_user_input = (
                f"The user is asking about a {st.session_state.plan_type.upper()} plan.\n\n"
                f"Here is some documentation context:\n{context}\n\n"
                f"User's question:\n{user_request}"
            )

            st.session_state.messages.append({"role": "user", "content": enriched_user_input})

            with st.spinner("Querying OpenRouter..."):
                response = query_openrouter(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)

    except Exception as e:
        st.error("Something went wrong with the OpenRouter request.")
        st.exception(e)

# Optional: display chat history in UI
with st.expander("Conversation history"):
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")