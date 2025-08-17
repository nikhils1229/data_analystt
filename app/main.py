from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
import sys

# --- Import processor safely ---
try:
    # Case 1: Running as package
    from .processor import process_question
except ImportError:
    # Case 2: Running as plain script
    from .processor import process_question

app = FastAPI()


@app.post("/api/")
async def analyze(
    questions_txt: UploadFile = File(..., alias="questions.txt"),
    data_csv: UploadFile = File(..., alias="data.csv"),
):
    # Read uploaded files
    questions_text = (await questions_txt.read()).decode("utf-8")
    file_contents = (await data_csv.read()).decode("utf-8")

    result = process(questions_text, file_contents)
    return result



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
