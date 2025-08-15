import io
import base64
import pandas as pd
from openai import OpenAI

client = OpenAI()

def process_question(files, question):
    # Step 1: Read files and combine into text
    file_texts = []
    for file in files:
        content = file.file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
            file_texts.append(df.to_csv(index=False))
        except Exception:
            try:
                df = pd.read_excel(io.BytesIO(content))
                file_texts.append(df.to_csv(index=False))
            except:
                file_texts.append(content.decode(errors="ignore"))

    combined_data = "\n\n".join(file_texts)

    # Step 2: Build universal prompt
    prompt = f"""
You are a world-class Python data analyst with matplotlib, pandas, and networkx.

TASK:
1. Read the dataset provided below.
2. Read the question carefully and extract the EXACT JSON key names and expected value types from it.
3. Perform the correct analysis in Python (you can create network graphs, histograms, bar charts, etc.).
4. If an image is required:
   - Save it as a PNG
   - Encode it in Base64
   - Ensure the encoded string is under 100 KB
5. DO NOT hallucinate values — compute them from the dataset.
6. The JSON keys and structure MUST exactly match what the question requests.
7. Output ONLY the JSON object, nothing else.

Dataset:
{combined_data}

Question:
{question}
"""

    # Step 3: Call LLM with JSON output enforced
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return response.choices[0].message["content"]
