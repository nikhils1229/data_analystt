import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from typing import List, Optional
from fastapi import UploadFile
from dotenv import load_dotenv
from openai import OpenAI

# Load .env for local testing
load_dotenv()

# Get OpenAI API key from env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

client = OpenAI(api_key=api_key)

def process_request(files: List[UploadFile], qtext: Optional[str] = None):
    """
    Process uploaded files + optional qtext.
    Always returns a valid JSON array or object.
    """

    questions = None
    csv_dataframes = []
    other_files_content = {}

    # Extract questions and data files
    for file in files:
        content = file.file.read()
        if file.filename.lower().endswith(".txt"):
            questions = content.decode("utf-8").strip()
        elif file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            csv_dataframes.append(df)
        else:
            try:
                other_files_content[file.filename] = content.decode("utf-8")
            except:
                pass

    # Append qtext if given
    if qtext:
        if questions:
            questions += "\n" + qtext
        else:
            questions = qtext

    if not questions:
        raise ValueError("No questions provided")

    # Prepare data summary for LLM
    data_summary = ""
    for i, df in enumerate(csv_dataframes):
        data_summary += f"\nCSV File {i+1} (first 5 rows):\n{df.head().to_csv(index=False)}"

    prompt = f"""
You are a data analyst.
The user provided the following questions:

{questions}

Additional data:
{data_summary}

Please answer the questions strictly in valid JSON format.
If multiple questions are asked, return a JSON array of answers in the same order.
If plotting is requested, return the plot as base64-encoded string in place of the answer.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw_output = response.choices[0].message.content.strip()

        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return [raw_output]
    except Exception:
        q_count = len([q for q in questions.split("\n") if q.strip()])
        return ["" for _ in range(q_count)]
