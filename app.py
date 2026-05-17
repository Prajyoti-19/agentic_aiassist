import streamlit as st
import re
import numpy as np
import pandas as pd
import os

from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import google.generativeai as genai

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# GEMINI LLM (ONLY FOR ANSWERS)
# -----------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm_model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# NLP (Preprocessing)
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', 
                  text)
    return text

# -----------------------------
# ML Model (Decision Making)
# -----------------------------
def train_model():
    data = [
        "summarize document",
        "find from file",
        "what is ai",
        "explain machine learning"
    ]
    labels = [1, 1, 0, 0]  # 1 = RAG, 0 = LLM

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)

    model = LogisticRegression()
    model.fit(X, labels)

    return model, vectorizer

# -----------------------------
# Load Documents
# -----------------------------
def load_docs():
    loader = TextLoader("sample.txt")
    return loader.load()

# -----------------------------
# Split Documents
# -----------------------------
def split_docs(docs):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# -----------------------------
# Create Vector DB (LOCAL EMBEDDINGS)
# -----------------------------
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embeddings)
    return db

# -----------------------------
# Retrieve Docs
# -----------------------------
def retrieve_docs(db, query):
    return db.similarity_search(query, k=3)

# -----------------------------
# Generate Answer (Gemini)
# -----------------------------
import time

def generate_answer(query, context):
    try:
        prompt = f"Context: {context}\nQuestion: {query}"
        response = llm_model.generate_content(prompt)
        return response.text

    except Exception as e:
        if "429" in str(e):
            time.sleep(40)  # wait as API says ~38 sec
            response = llm_model.generate_content(prompt)
            return response.text

        return f"Error: {str(e)}"
# -----------------------------
# Agent Decision
# -----------------------------
def agent_decision(query, model, vectorizer):
    X = vectorizer.transform([query])
    pred = model.predict(X)
    return "RAG" if pred[0] == 1 else "LLM"

# -----------------------------
# Initialize System (RUN ONCE)
# -----------------------------
docs = load_docs()
chunks = split_docs(docs)
db = create_vectorstore(chunks)
ml_model, vectorizer = train_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Agentic AI Assistant")

query = st.text_input("Ask something:")

if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a question")
    else:
        query_clean = clean_text(query)

        decision = agent_decision(query_clean, ml_model, vectorizer)

        if decision == "RAG":
            retrieved = retrieve_docs(db, query_clean)
            context = " ".join([doc.page_content for doc in retrieved])
        else:
            context = ""

        answer = generate_answer(query_clean, context)

        st.write(" Decision:", decision)
        st.write(" Answer:")
        st.write(answer)

# -----------------------------
# Logging with Pandas
# -----------------------------
if "logs" not in st.session_state:
    st.session_state.logs = pd.DataFrame(columns=["query", "decision", "answer"])

if st.button("Save Last Result"):
    if "decision" in locals():
        new_row = {
            "query": query,
            "decision": decision,
            "answer": answer
        }
        st.session_state.logs = pd.concat(
            [st.session_state.logs, pd.DataFrame([new_row])],
            ignore_index=True
        )

        st.success("Saved!")
        st.dataframe(st.session_state.logs)