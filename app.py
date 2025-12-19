import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

HF_API_KEY = st.secrets["HF_API_KEY"]                #Hugging Face LLM Config

HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "google/flan-t5-base"
)

HF_HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

#Page setup

st.set_page_config("Data-to-Insights RAG Agent", layout="wide")    
st.title("Data-to-Insights RAG Agent")
st.caption("Deterministic Analytics + Agentic RAG (Hugging Face LLM)")


file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])            #Upload CSV / Excel
if not file:
    st.stop()

df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)


df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]          # Normalize column names

#Basic Data Cleaning

for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).fillna("unknown")
    else:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(axis=1, how="all")

#Schema inference

def infer_schema(df):
    schema = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            schema[c] = "numeric"
        elif df[c].nunique() / max(len(df), 1) > 0.9:
            schema[c] = "id"
        else:
            schema[c] = "category"
    return schema

schema = infer_schema(df)
numeric_cols = [c for c, v in schema.items() if v == "numeric"]
category_cols = [c for c, v in schema.items() if v == "category"]


#EDA

st.subheader("Dataset Preview")
st.dataframe(df.head())
st.write("Shape:", df.shape)

with st.expander("EDA Summary"):
    st.write("**Numeric columns:**", numeric_cols)
    st.write("**Categorical columns:**", category_cols)


#Embedding Model (Local)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
col_embeddings = embedder.encode(df.columns.tolist())

def best_column(query, cols):
    if not cols:
        return None
    q_emb = embedder.encode([query])
    idxs = [df.columns.get_loc(c) for c in cols]
    sims = cosine_similarity(q_emb, col_embeddings[idxs])[0]
    return cols[int(np.argmax(sims))]


#RAG: Row Embeddings

df["row_text"] = df.astype(str).agg(" | ".join, axis=1)
row_embeddings = embedder.encode(df["row_text"].tolist())

def retrieve_rows(query, k=5):
    q_emb = embedder.encode([query])
    sims = cosine_similarity(q_emb, row_embeddings)[0]
    top_idx = np.argsort(sims)[-k:][::-1]
    return df.iloc[top_idx]


#Analytics Agent(Deterministic)

def compute_answer(query):
    q = query.lower()

    if "how many" in q or "number of" in q:
        col = best_column(query, category_cols)
        if col:
            return f"{df[col].nunique()} unique {col}"

    if "average" in q or "mean" in q:
        col = best_column(query, numeric_cols)
        if col:
            return f"Average {col}: {df[col].mean():.2f}"

    if "total" in q or "sum" in q:
        col = best_column(query, numeric_cols)
        if col:
            return f"Total {col}: {df[col].sum():.2f}"

    if "highest" in q or "maximum" in q:
        col = best_column(query, numeric_cols)
        if col:
            return f"Highest {col}: {df[col].max():.2f}"

    if "lowest" in q or "minimum" in q:
        col = best_column(query, numeric_cols)
        if col:
            return f"Lowest {col}: {df[col].min():.2f}"

    return None


#Insight Agent(LLM)

def call_llm(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 256}
    }

    r = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json=payload,
        timeout=60
    )

    if r.status_code != 200:
        return "LLM temporarily unavailable. Showing deterministic result only."

    result = r.json()

    if isinstance(result, list) and result:
        return result[0].get("generated_text", "")
    elif isinstance(result, dict):
        return result.get("generated_text", "")
    else:
        return "Empty LLM response."

def generate_insight(question, computed_answer, retrieved_rows):
    context = "\n".join(retrieved_rows["row_text"].tolist())

    prompt = f"""
You are a business analytics AI assistant.

Question:
{question}

Deterministic Computed Answer (DO NOT CHANGE NUMBERS):
{computed_answer}

Relevant Data Rows:
{context}

Explain the result in simple business terms.
Highlight trends and implications.
Do NOT invent new numbers.
"""

    return call_llm(prompt)


#Q&A Interface

st.markdown("---")
st.subheader("Ask a Business Question")

query = st.text_input(
    "Examples: How many employees are there? | What is the highest sales amount?"
)

if query:
    computed = compute_answer(query)
    if not computed:
        st.error("Unsupported analytical question.")
    else:
        rows = retrieve_rows(query)
        insight = generate_insight(query, computed, rows)
        st.success(computed)
        st.markdown("Insights") #LLM insight RAG powered
        st.write(insight)


#Chart Builder

st.markdown("---")
st.subheader("Chart Builder")

if category_cols and numeric_cols:
    x = st.selectbox("X-axis (category)", category_cols)
    y = st.selectbox("Y-axis (numeric)", numeric_cols)

    if st.button("Generate Chart"):
        grouped = df.groupby(x)[y].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        grouped.plot(kind="bar", ax=ax)
        ax.set_title(f"{y} by {x}")
        plt.tight_layout()
        st.pyplot(fig)


#Executive Insight Report

st.markdown("---")
st.subheader("Executive Insight Report")

if st.button("Generate Report"):
    st.write(f"Dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
    st.dataframe(df[numeric_cols].describe())
