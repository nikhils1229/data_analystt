import os
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_chart():
    """Helper to capture current matplotlib figure as base64 PNG."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def analyze_csv_generic(csv_file: str, questions: str):
    """
    Generic CSV analyzer that:
    1. Computes numeric + categorical summaries.
    2. Builds automatic visualizations.
    3. Uses OpenAI to answer custom questions.
    """
    df = pd.read_csv(csv_file)

    results = {}

    # --- Basic stats ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if len(numeric_cols) > 0:
        for col in numeric_cols:
            results[f"{col}_sum"] = df[col].sum()
            results[f"{col}_mean"] = float(df[col].mean())
            results[f"{col}_median"] = float(df[col].median())
            results[f"{col}_min"] = float(df[col].min())
            results[f"{col}_max"] = float(df[col].max())

        # Correlation matrix (store only top correlations to save space)
        corr = df[numeric_cols].corr().to_dict()
        results["correlations"] = corr

        # Histogram of first numeric column
        plt.hist(df[numeric_cols[0]], bins=10, color="blue")
        plt.title(f"Histogram of {numeric_cols[0]}")
        results["histogram_chart"] = encode_chart()

    if len(cat_cols) > 0:
        for col in cat_cols:
            top_val = df[col].mode()[0] if not df[col].mode().empty else None
            results[f"{col}_mode"] = str(top_val)
            freq = df[col].value_counts().to_dict()
            results[f"{col}_frequencies"] = freq

        # If categorical + numeric → bar chart
        if numeric_cols:
            plt.bar(df[cat_cols[0]], df[numeric_cols[0]], color="green")
            plt.xticks(rotation=45)
            plt.title(f"{numeric_cols[0]} by {cat_cols[0]}")
            results["bar_chart"] = encode_chart()

    # --- Try to detect date/time for line chart ---
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if date_cols and numeric_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        df_sorted = df.dropna(subset=[date_cols[0]]).sort_values(by=date_cols[0])
        if not df_sorted.empty:
            plt.plot(df_sorted[date_cols[0]], df_sorted[numeric_cols[0]], color="red")
            plt.title(f"{numeric_cols[0]} over {date_cols[0]}")
            results["line_chart"] = encode_chart()

    # --- LLM reasoning for custom Q&A ---
    try:
        summary = df.describe(include="all").to_string()
        prompt = f"""
You are a data analyst.
Dataset summary:
{summary}

Questions:
{questions}

Answer clearly in JSON format.
"""
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "system", "content": "You are a data analyst."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        llm_ans = resp.choices[0].message.content
        results["llm_answers"] = llm_ans
    except Exception as e:
        results["llm_error"] = str(e)

    return results
