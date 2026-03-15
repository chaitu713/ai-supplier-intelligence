# Supplier AI System

This project is an AI-driven Supplier Intelligence prototype that predicts supplier risk and analyzes ESG (Environmental, Social, Governance) metrics using Azure OpenAI. It provides a FastAPI backend and a Streamlit dashboard frontend.

## Features
- Load supplier and ESG datasets (CSV)
- Train a machine learning model to predict supplier risk
- Use Azure OpenAI to analyze ESG risk and generate explanations
- REST API backend (FastAPI)
- Web dashboard frontend (Streamlit)

## Project Structure
```
supplier-ai-system/
│
├── data/
│   ├── suppliers.csv
│   ├── esg_metrics.csv
│   └── transactions.csv
│
├── backend/
│   ├── data_generator.py
│   ├── risk_model.py
│   ├── ai_agent.py
│   ├── database.py
│   └── api.py
│
├── frontend/
│   └── app.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Azure OpenAI environment variables**

   Set the following environment variables with your Azure OpenAI credentials:
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_KEY`

3. **Run the backend API**

   ```bash
   cd backend
   uvicorn api:app --reload
   ```

4. **Run the frontend dashboard**

   In a new terminal:
   ```bash
   cd frontend
   streamlit run app.py
   ```

5. **Explore the Jupyter notebook**

   Open `notebooks/exploration.ipynb` in Jupyter for data exploration and prototyping.

## Notes
- The backend must be running before using the Streamlit dashboard.
- Replace sample data and API keys with your actual data and credentials.
