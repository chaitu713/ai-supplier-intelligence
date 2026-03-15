import pandas as pd

############################################################

# MAIN AI AGENT FUNCTION

############################################################

def handle_question(question, performance_df, suppliers_df):

    question = question.lower()

    ########################################################
    # QUESTION 1
    ########################################################

    if "risky" in question and "which" in question:

        risky_suppliers = performance_df.sort_values(
            "risk_score", ascending=False
        ).head(5)

        explanation = (
            "Based on supplier operational performance metrics such as "
            "delivery delays, defect rates, and cost variance, the following "
            "suppliers show elevated operational risk."
        )

        return explanation, risky_suppliers

    ########################################################
    # QUESTION 2
    ########################################################

    if "why" in question:

        supplier = performance_df.sort_values(
            "risk_score", ascending=False
        ).iloc[0]

        explanation = f"""
Supplier **{supplier['supplier_name']}** shows elevated risk due to the following factors:

• Average delivery delay: **{round(supplier['avg_delay'], 2)} days**
• Product defect rate: **{round(supplier['avg_defect'] * 100, 2)}%**
• Cost variance: **{round(supplier['avg_cost_variance'], 2)}%**

These factors indicate potential supply chain reliability issues and may impact production planning.
"""

        return explanation, None

    ########################################################
    # QUESTION 3
    ########################################################

    if "recommend" in question:

        recommendations = performance_df.sort_values(
            "risk_score"
        ).head(5)

        explanation = (
            "Based on supplier performance analytics, the following "
            "suppliers are recommended alternatives with lower operational risk."
        )

        return explanation, recommendations

    ########################################################
    # DEFAULT RESPONSE
    ########################################################

    return (
        "Try asking one of the following questions:\n"
        "• Which suppliers are risky?\n"
        "• Why this supplier is risky?\n"
        "• Recommend alternate suppliers",
        None
    )
