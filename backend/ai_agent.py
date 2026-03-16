from google import genai
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)


def ask_supplier_ai(question, performance_df):
    """
    Sends supplier performance data + user question to Gemini
    and returns AI-generated insights.
    """

    try:

        # Sort suppliers by risk to prioritize important data
        context_data = performance_df.sort_values(
            "risk_score", ascending=False
        )[[
            "supplier_name",
            "country",
            "category",
            "avg_delay",
            "avg_defect",
            "avg_cost_variance",
            "risk_score"
        ]].head(50)

        # Convert dataframe to text
        data_text = context_data.to_string(index=False)

        # Prompt for Gemini
        prompt = f"""
You are an AI procurement assistant.

Supplier dataset:
{data_text}

User Question:
{question}

Instructions:
- Answer ONLY the user question.
- Maximum 4 bullet points.
- Maximum 5 suppliers if listing.
- Be concise and direct.
"""

        # Call Gemini
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Error generating AI response: {str(e)}"