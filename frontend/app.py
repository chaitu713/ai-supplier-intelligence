import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from backend.blob_storage import upload_file_to_blob
from backend.document_intelligence import extract_document
from backend.data_append import process_extracted_document
from backend.document_history import log_document
from backend.ai_agent import handle_question


st.set_page_config(page_title="AI Supplier Intelligence", layout="wide")

############################################################
# HEADER
############################################################

# st.title("TCS Envirozoneᴬᴵ 4.0")

st.markdown(
"""
# TCS Envirozone<sup>AI</sup> 4.0

## Responsible Sourcing & Supplier Intelligence

### AI-Driven Supplier Risk Monitoring & ESG Analytics  

Upload supplier documents, extract operational and ESG data using AI, and monitor supplier performance through intelligent analytics.
""",
unsafe_allow_html=True
)

############################################################
# LOAD DATA
############################################################

def load_data():
    suppliers = pd.read_csv("data/suppliers.csv")
    esg = pd.read_csv("data/esg_metrics.csv")
    transactions = pd.read_csv("data/transactions.csv")
    return suppliers, esg, transactions


suppliers, esg, transactions = load_data()

############################################################
# SIDEBAR NAVIGATION
############################################################

st.sidebar.title("Navigation")

page = st.sidebar.radio(
"",
[
"📄 Document Ingestion",
"🔎 Supplier Explorer",
"📊 Overview Dashboard",
"⚠ Risk Monitoring",
"🤖 AI Insights"
]
)

############################################################
# SIDEBAR STATUS
############################################################

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

st.sidebar.metric("Suppliers", len(suppliers))
st.sidebar.metric("Transactions", len(transactions))

history_df = pd.read_csv("data/document_history.csv")
st.sidebar.metric("Documents Processed", len(history_df))

############################################################
# BUILD PERFORMANCE DATASET
############################################################

performance = transactions.groupby("supplier_id").agg(
    avg_delay=("delivery_delay_days", "mean"),
    avg_defect=("defect_rate", "mean"),
    avg_cost_variance=("cost_variance", "mean")
).reset_index()

performance["risk_score"] = (
    performance["avg_delay"] * 0.4
    + performance["avg_defect"] * 100 * 0.4
    + abs(performance["avg_cost_variance"]) * 0.2
)

performance = performance.merge(
    suppliers[["supplier_id", "supplier_name", "country", "category"]],
    on="supplier_id"
)

############################################################
# DOCUMENT INGESTION
############################################################

if page == "📄 Document Ingestion":

    st.subheader("Supplier Document Upload")

    col1, col2, col3 = st.columns(3)

    with col1:
        supplier_file = st.file_uploader(
            "Supplier Onboarding Document", type=["pdf"]
        )

    with col2:
        esg_file = st.file_uploader(
            "ESG Assessment Report", type=["pdf"]
        )

    with col3:
        transaction_file = st.file_uploader(
            "Transaction Report", type=["pdf"]
        )

    if supplier_file and esg_file and transaction_file:

        if st.button("🚀 Process Documents"):

            os.makedirs("uploads", exist_ok=True)

            with st.spinner("Running ingestion pipeline..."):

                supplier_path = os.path.join("uploads", supplier_file.name)
                esg_path = os.path.join("uploads", esg_file.name)
                transaction_path = os.path.join("uploads", transaction_file.name)

                with open(supplier_path, "wb") as f:
                    f.write(supplier_file.read())

                with open(esg_path, "wb") as f:
                    f.write(esg_file.read())

                with open(transaction_path, "wb") as f:
                    f.write(transaction_file.read())

                st.success("Documents staged")

                supplier_blob = upload_file_to_blob(
                    supplier_path, supplier_file.name
                )
                esg_blob = upload_file_to_blob(
                    esg_path, esg_file.name
                )
                transaction_blob = upload_file_to_blob(
                    transaction_path, transaction_file.name
                )

                supplier_text = extract_document(supplier_blob)
                esg_text = extract_document(esg_blob)
                transaction_text = extract_document(transaction_blob)

                doc_type1, count1 = process_extracted_document(supplier_text)
                log_document(supplier_file.name, doc_type1, count1)

                doc_type2, count2 = process_extracted_document(esg_text)
                log_document(esg_file.name, doc_type2, count2)

                doc_type3, count3 = process_extracted_document(transaction_text)
                log_document(transaction_file.name, doc_type3, count3)

                st.success("Ingestion pipeline completed")

    st.markdown("---")

    st.subheader("Document Processing History")

    history = pd.read_csv("data/document_history.csv")
    history = history.sort_values("timestamp", ascending=False)

    st.dataframe(history, use_container_width=True)

############################################################
# SUPPLIER EXPLORER
############################################################

elif page == "🔎 Supplier Explorer":

    st.subheader("Supplier Dataset Explorer")

    dataset = st.selectbox(
        "Choose Dataset",
        ["Suppliers", "ESG Metrics", "Transactions"]
    )

    if dataset == "Suppliers":
        st.dataframe(suppliers, use_container_width=True)

    elif dataset == "ESG Metrics":
        st.dataframe(esg, use_container_width=True)

    else:
        st.dataframe(transactions, use_container_width=True)

############################################################
# OVERVIEW DASHBOARD
############################################################

elif page == "📊 Overview Dashboard":

    st.subheader("Supplier Performance Overview")

    total_suppliers = suppliers.shape[0]
    avg_esg_score = round(esg["esg_score"].mean(), 2)
    avg_delay = round(transactions["delivery_delay_days"].mean(), 2)
    avg_defect = round(transactions["defect_rate"].mean(), 3)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Suppliers", total_suppliers)
    col2.metric("Avg ESG Score", avg_esg_score)
    col3.metric("Avg Delivery Delay", avg_delay)
    col4.metric("Avg Defect Rate", avg_defect)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Supplier Distribution by Country")
        st.bar_chart(suppliers["country"].value_counts())

    with col2:
        st.subheader("ESG Score Distribution")

        fig, ax = plt.subplots()
        ax.hist(esg["esg_score"], bins=20)
        st.pyplot(fig)

############################################################
# RISK MONITORING
############################################################

elif page == "⚠ Risk Monitoring":

    st.subheader("Supplier Risk Monitoring")

    high_risk = performance[performance["risk_score"] > 8]
    high_risk = high_risk.sort_values("risk_score", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("High Risk Suppliers")
        st.dataframe(high_risk, use_container_width=True)

    with col2:
        st.subheader("Top Risk Scores")
        st.bar_chart(high_risk.set_index("supplier_name")["risk_score"].head(10))

############################################################
# AI INSIGHTS COPILOT
############################################################

elif page == "🤖 AI Insights":

    st.subheader("Supplier Intelligence Copilot")

    st.markdown(
        """
Suggested Questions

• Which suppliers are risky?  
• Why this supplier is risky?  
• Recommend alternate suppliers
"""
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask about suppliers...")

    if question:

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        explanation, table = handle_question(
            question,
            performance,
            suppliers
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": explanation}
        )

        with st.chat_message("assistant"):
            st.write(explanation)

            if table is not None:
                st.dataframe(table, use_container_width=True)

############################################################
# FOOTER
############################################################

st.markdown("---")
st.caption("AI Supplier Intelligence Platform | Built with Azure AI")