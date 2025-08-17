from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
import sys

# --- Import processor safely ---
try:
    # Case 1: Running as package
    from .processor import process
except ImportError:
    # Case 2: Running as plain script
    from processor import process

app = FastAPI()


@app.post("/api/")
async def analyze(
    questions: UploadFile = File(..., alias="questions.txt"),
    data: UploadFile = File(..., alias="data.csv"),
):
    # Read uploaded files
    questions_text = (await questions.read()).decode("utf-8")
    file_contents = (await data.read()).decode("utf-8")

    # Call processor
    try:
        result = process(questions_text, file_contents)
    except Exception as e:
        return {"error": str(e)}

    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
