import os
import json
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Try importing new OpenAI SDK
try:
    from openai import OpenAI
    NEW_OPENAI = True
except ImportError:
    import openai
    NEW_OPENAI = False

# Initialize client
if NEW_OPENAI:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")

def process_question(question: str, files: list):
    """
    Generic LLM-driven analysis.
    Reads CSVs, understands the question, performs calculations/plots,
    and returns a structured JSON answer.
    """

    dfs = {}
    for file in files:
        df = pd.read_csv(file.file)
        dfs[file.filename] = df

    # Describe available data
    description_parts = []
    for name, df in dfs.items():
        description_parts.append(f"File: {name}\nColumns: {list(df.columns)}\nSample:\n{df.head(3).to_dict()}")

    prompt = f"""
You are a data analyst.
You are given the following datasets:
{chr(10).join(description_parts)}

Question:
{question}

If the question involves network/graph analysis, output JSON with:
- edge_count
- highest_degree_node
- average_degree
- density
- shortest_path_alice_eve
- network_graph (base64 PNG under 100kB)
- degree_histogram (base64 PNG under 100kB)

If it’s another type of analysis, answer in JSON format with all necessary keys and values.
Do not include explanations — only output valid JSON.
"""

    # Query LLM
    if NEW_OPENAI:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_output = response.choices[0].message.content
    else:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_output = response.choices[0].message["content"]

    try:
        result = json.loads(raw_output)
    except:
        result = {"error": "Invalid JSON from model", "raw_output": raw_output}

    return result
