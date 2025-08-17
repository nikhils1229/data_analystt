from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from .processor import process_question
import uvicorn

app = FastAPI()

@app.post("/api/")
async def analyze(
    questions_txt: UploadFile = File(..., alias="questions.txt"),   # match grader field
    files: List[UploadFile] = File(default=[])
):
    try:
        # Read the natural language question(s) from questions.txt
        question_text = (await questions_txt.read()).decode("utf-8")

        # Pass question + any attached files (CSV, etc.)
        result = process_question(question_text, files)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
