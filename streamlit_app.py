import streamlit as st
from openai import OpenAI
import os

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Streamlit setup ---
st.set_page_config(page_title="Ata-transform", page_icon="images/ataccama_logo.png")
st.title("Welcome to :violet[Ataccama's] Transformation Assistant Bot")

# --- Load and cache vector stores ---
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

# --- OpenRouter integration ---
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

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are Ata-Cat, an expert in Ataccama data transformation logic. Respond clearly and helpfully, and sound conversational and friendly."
        }
    ]

if "plan_type" not in st.session_state:
    st.session_state.plan_type = None

if "pending_request" not in st.session_state:
    st.session_state.pending_request = None

desktop_retriever, webapp_retriever = load_vectorstores()

# --- Helpers ---
def detect_plan_type(text):
    text = text.lower()
    if "desktop" in text or "one desktop" in text:
        return "desktop"
    elif "webapp" in text or "transformation plan" in text:
        return "webapp"
    return None

def get_contextual_docs(user_input):
    detected = detect_plan_type(user_input)

    # Update plan type if specified
    if detected:
        st.session_state.plan_type = detected

    # Use session-stored plan type to fetch docs
    if st.session_state.plan_type == "desktop":
        return desktop_retriever.get_relevant_documents(user_input), "desktop"
    elif st.session_state.plan_type == "webapp":
        return webapp_retriever.get_relevant_documents(user_input), "webapp"
    else:
        return [], "ambiguous"

# --- UI ---
st.write("Hi! I'm :violet[Ata-cat]. What can I help you build today?")
user_request = st.chat_input("Type your question or say hello...")

if user_request:
    context_docs, context_type = get_contextual_docs(user_request)

    # Handle ambiguity
    if context_type == "ambiguous" and st.session_state.plan_type is None:
        st.session_state.pending_request = user_request
        ask = "Got it! Just to help me retrieve the right info: is this for a **ONE Desktop** plan or a **WebApp transformation** plan?"
        st.session_state.messages.append({"role": "assistant", "content": ask})
        st.markdown(f"**You asked:** {user_request}")
        st.markdown(ask)

    else:
        actual_request = user_request if st.session_state.pending_request is None else st.session_state.pending_request
        st.session_state.pending_request = None

        enriched_user_input = actual_request

        # Add relevant docs if plan_type is known
        if st.session_state.plan_type in ("desktop", "webapp"):
            context = "\n\n".join([doc.page_content for doc in context_docs[:2]])
            enriched_user_input = (
                f"The user is asking about a {st.session_state.plan_type.upper()} plan.\n\n"
                f"Here is some documentation context:\n{context}\n\n"
                f"User's question:\n{actual_request}"
            )

        # Append user message (with or without context)
        st.session_state.messages.append({"role": "user", "content": enriched_user_input})

        st.markdown(f"**You asked:** {actual_request}")

        with st.spinner("Let me check the docs for you..."):
            response = query_openrouter(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

# --- Conversation history ---
with st.expander("Conversation history"):
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")