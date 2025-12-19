# Data-to-Insights RAG Agent

A Streamlit-based analytics assistant that allows users to upload CSV or Excel files and ask business questions in natural language.

This project combines **deterministic data analytics** with a **Retrieval-Augmented Generation (RAG)** approach to provide accurate numerical results along with simple, human-readable explanations.

## Project Overview

Most business data exists in spreadsheets, but extracting insights often requires technical skills or BI tools.  
This application enables non-technical users to upload tabular data and ask questions like *“What is the total sales?”* or *“How many employees are there?”* in plain English.

All numerical calculations are performed using Pandas, while an LLM is used **only to explain results**, not to compute them.

## Features

- Upload and preview CSV / Excel files
- Automatic detection of numeric and categorical columns
- Natural language question answering using RAG
- Deterministic analytics (count, sum, average, min, max)
- Simple chart generation
- Executive summary statistics

## Tech Stack

- **Frontend / UI:** Streamlit  
- **Data Processing:** Pandas, NumPy  
- **Visualisation:** Matplotlib  
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)  
- **LLM:** Hugging Face Inference API (FLAN-T5)  
- **Retrieval:** Cosine similarity on row-level embeddings  


## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/shraddhasoudagar27-ui/data-to-insights-rag-agent.git
   cd data-to-insights-rag-agent

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Add your Hugging Face API key:
   Create a file at:
   ```bash
   .streamlit/secrets.toml

 Add:
 
    HF_API_KEY = "your_huggingface_api_key"

6. Run the application:
   ```bash
   streamlit run app.py

### Sample Dataset

The repository includes a sample CSV (sample.csv) with the following structure:
```bash
   employee_id,employee_name,department,region,month,sales_amount,units_sold
   E001,Alice,Sales,North,Jan,12000,40
   E002,Bob,Sales,South,Jan,9500,35
   E003,Charlie,Marketing,North,Jan,7000,20
   E004,Diana,HR,West,Jan,4000,10
   E005,Evan,Sales,East,Feb,15000,50
   E006,Fiona,Marketing,South,Feb,8200,22
   E007,George,Sales,North,Feb,11000,38
   E008,Hannah,HR,East,Feb,4200,12
   E009,Ian,Sales,West,Mar,17000,55
   E010,Jane,Marketing,North,Mar,9000,25
   E011,Kevin,Sales,South,Mar,13000,45
   E012,Laura,HR,West,Mar,3900,9
```

### Example Questions You Can Ask

Using the sample dataset, users can ask questions such as:
What is the total units sold?
What is the average sales amount?
What is the highest sales amount?
How many employees are there?

The application computes the numeric answer deterministically and then explains the result using relevant data rows.

### How It Works (High Level)

Data is uploaded and lightly cleaned
Column schema is inferred automatically
Numeric answers are computed using Pandas
Relevant rows are retrieved using semantic similarity (RAG)
The LLM explains the computed result in simple terms

### Notes

All numerical calculations are performed locally
The LLM is used only for explanation, not for computation
This design helps avoid incorrect or hallucinated numbers

### Future Enhancements (Not Implemented)

Support for multiple datasets and joins
Date-aware and time-series analytics
Richer interactive dashboards
Authentication and role-based access control
