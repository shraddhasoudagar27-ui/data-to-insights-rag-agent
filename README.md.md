# Data-to-Insights RAG Agent

A Streamlit-based GenAI analytics assistant that allows users to upload CSV/Excel files and ask business questions using natural language.

## Features
- Deterministic analytics using Pandas
- Retrieval-Augmented Generation (RAG) with local embeddings
- LLM-based explanations using Hugging Face (FLAN-T5)
- Simple chart generation and summary reports

## How to Run Locally

1. Install dependencies:
pip install -r requirements.txt


2. Add Hugging Face API key:
Create `.streamlit/secrets.toml`

HF_API_KEY = "your_huggingface_api_key"


3. Run the app:
streamlit run app.py


## Notes
- All numerical calculations are performed locally.
- The LLM is used only for explanation, not computation.
