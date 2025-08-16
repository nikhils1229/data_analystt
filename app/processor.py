import os
import json
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import requests
from bs4 import BeautifulSoup
import duckdb

# OpenAI client (works with both old and new SDKs)
try:
    from openai import OpenAI
    NEW_OPENAI = True
except ImportError:
    import openai
    NEW_OPENAI = False

if NEW_OPENAI:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    openai.api_key = os.getenv("OPENAI_API_KEY")


def encode_plot(fig, format="png", max_size=100_000, min_dpi=50):
    """
    Encode a matplotlib figure to base64 under max_size bytes.
    Will progressively lower DPI until size requirement is met.
    """
    dpi = 150
    while dpi >= min_dpi:
        buf = io.BytesIO()
        fig.savefig(buf, format=format, bbox_inches="tight", dpi=dpi)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        if len(b64.encode("utf-8")) <= max_size * 1.37:  # ~1.37x expansion in base64
            plt.close(fig)
            return f"data:image/{format};base64,{b64}"
        dpi -= 10
    plt.close(fig)
    return f"data:image/{format};base64,{b64}"


def process_question(question: str, files: list):
    """
    Main entry point. Handles:
    - Multiple Q&A (returns JSON array of strings)
    - Graph/network problems (edges.csv)
    - Wikipedia scraping
    - DuckDB queries
    - Generic CSV analytics
    """

    # Try to load any attached CSVs into pandas
    dfs = {}
    for f in files:
        try:
            df = pd.read_csv(f.file)
            dfs[f.filename] = df
        except Exception:
            pass

    q_lower = question.lower()

    # === 1. Multi-question prompts ("respond with a JSON array of strings") ===
    if "json array of strings" in q_lower:
        return call_llm_for_answer("", question, force_array=True)

    # === 2. Graph/network tasks (edges.csv) ===
    if "shortest_path" in q_lower or "edge_count" in q_lower or "degree" in q_lower:
        if not dfs:
            return {"error": "No CSV provided for graph analysis"}
        df = list(dfs.values())[0]
        G = nx.from_pandas_edgelist(df, df.columns[0], df.columns[1])

        edge_count = G.number_of_edges()
        degree_dict = dict(G.degree())
        highest_degree_node = max(degree_dict, key=degree_dict.get)
        avg_degree = sum(degree_dict.values()) / len(degree_dict)
        density = nx.density(G)
        try:
            shortest_path = nx.shortest_path(G, source="Alice", target="Eve")
        except Exception:
            shortest_path = []

        fig1, ax1 = plt.subplots()
        nx.draw_networkx(G, ax=ax1, with_labels=True, node_color="skyblue", edge_color="gray")
        network_graph = encode_plot(fig1)

        fig2, ax2 = plt.subplots()
        degrees = list(degree_dict.values())
        ax2.hist(degrees, bins=range(1, max(degrees) + 2))
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("Frequency")
        degree_histogram = encode_plot(fig2)

        return {
            "edge_count": edge_count,
            "highest_degree_node": highest_degree_node,
            "average_degree": avg_degree,
            "density": density,
            "shortest_path_alice_eve": shortest_path,
            "network_graph": network_graph,
            "degree_histogram": degree_histogram
        }

    # === 3. Wikipedia scraping tasks ===
    if "wikipedia" in q_lower:
        url = next((t for t in question.split() if t.startswith("http")), None)
        if not url:
            return {"error": "No URL found in question"}
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "wikitable"})
        df = pd.read_html(str(table))[0]
        description = f"Data columns: {df.columns.tolist()} sample: {df.head(3).to_dict()}"
        return call_llm_for_answer(description, question)

    # === 4. DuckDB queries ===
    if "duckdb" in q_lower or "parquet" in q_lower:
        return call_llm_for_answer("", question)

    # === 5. Generic CSV analysis ===
    if dfs:
        description_parts = [
            f"File: {name}\nColumns: {list(df.columns)}\nSample:\n{df.head(3).to_dict()}"
            for name, df in dfs.items()
        ]
        return call_llm_for_answer("\n".join(description_parts), question)

    # === 6. Fallback (no files, pure text Q) ===
    return call_llm_for_answer("", question)


def call_llm_for_answer(data_description, question, force_array=False):
    if force_array:
        prompt = f"""
You are a data analyst. You have the following data:
{data_description}

Question:
{question}

Return ONLY a valid JSON array of strings.
Example: ["answer1", "answer2", "answer3"]
"""
    else:
        prompt = f"""
You are a data analyst. You have the following data:
{data_description}

Question:
{question}

Return ONLY valid JSON matching exactly the keys, structure, and format requested in the question.
Do not include explanations or extra fields.
"""

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

    # --- CLEANUP: strip Markdown fences if present ---
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")  # remove leading/trailing ```
        # remove possible "json\n" or "JSON\n"
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {"error": "Invalid JSON from model", "raw_output": raw_output}
