import streamlit as st
import openai as ai
from bs4 import BeautifulSoup
import requests as req
import pandas as pd
import os

from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Streamlit setup
st.set_page_config(page_title="Ata-transform-bot")
st.title("Welcome to Ataccama's Transformation Assistant Bot")

#ai.api_key = st.secrets["OpenAI_key"]

# -------- Cached data loader and vector store builder --------
@st.cache_resource
def load_vectorstore():
    docu_url = 'https://docs.ataccama.com/one/latest/data-quality/data-transformation-plans.html'
    expressions_url = "https://docs.ataccama.com/one/latest/common-actions/one-expressions.html"

    docu_response = req.get(docu_url).text
    expressions_response = req.get(expressions_url).text

    bsoup = BeautifulSoup(docu_response, 'html.parser')
    bsoupExpressions = BeautifulSoup(expressions_response, 'html.parser')

    main_content = bsoup.find("main")
    main_content_expressions = bsoupExpressions.find("main")
    main_content_expressions_non_table = bsoupExpressions.find("main")

    for tag in main_content_expressions_non_table(["table", "script", "style", "nav", "aside"]):
        tag.decompose()

    paragraphs = main_content_expressions_non_table.find_all(text=True)
    non_table_text = "\n".join([p.strip() for p in paragraphs if p.strip()])

    output_lines = []
    tables = main_content_expressions.find_all("table")
    for i, table in enumerate(tables):
        df = pd.read_html(str(table))[0]
        table_text = df.to_string(index=False)
        output_lines.append(f"\n\n--- Table {i+1} ---\n{table_text}")

    os.makedirs("docs", exist_ok=True)

    with open("docs/ata-expressions-tables.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    if not main_content:
        main_content = bsoup.find("div", id="main") or bsoup.find("div", class_="main")
        main_content_expressions = bsoupExpressions.find("div", id="main") or bsoupExpressions.find("div", class_="main")

    if main_content:
        docutext = main_content.get_text(separator="\n", strip=True)
        expressiontext = main_content_expressions.get_text(separator="\n", strip=True)

        with open("docs/ata-documentation.txt", "w") as f:
            f.write(docutext)

        with open("docs/ata-documentation-expressions.txt", "w") as f:
            f.write(expressiontext)

    with open("docs/expression_information_non_tabular.txt", "w", encoding="utf-8") as f:
        f.write(non_table_text)

    file_paths = [
        "docs/ata-documentation.txt",
        "docs/ata-documentation-expressions.txt",
        "docs/expression_information_non_tabular.txt",
        "docs/consultant_examples_repo.txt"
    ]

    documents = []
    for file_path in file_paths:
        loader = TextLoader(file_path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    if os.path.exists("vector_index_chunks"):
        db = FAISS.load_local("vector_index_chunks", embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local("vector_index_chunks")

    return db.as_retriever()

# Load retriever once
retriever = load_vectorstore()

# -------- UI --------
st.write("Let's start building! What can we help you build today?")
user_request = st.chat_input("Please enter a transformation plan request")

def is_ollama_running():
    try:
        r = req.get("http://localhost:11434")
        return r.status_code == 200
    except:
        return False

# -------- Response Logic --------
if user_request:
    if not is_ollama_running():
        st.error("Ollama is not running. Start it with: `ollama serve`")
    else:
        try:
            llm = Ollama(model="llama3")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )
            with st.spinner("Thinking..."):
                gpt_response = qa_chain.run(user_request)
                st.write(gpt_response)
        except Exception as e:
            st.error("Something went wrong.")
            st.exception(e)