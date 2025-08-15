from fastapi import FastAPI, UploadFile, Form, File, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
from .processor import process_question
import uvicorn

app = FastAPI()

@app.post("/api/")
async def analyze(
    request: Request,
    question: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[])
):
    try:
        # Case 1: If content type is JSON
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            question = data.get("question")
            # Here you'd need to handle files if they're base64 or URLs in JSON
            files = []  # adjust if JSON files are passed differently

        # Validation
        if not question:
            return JSONResponse(
                content={"error": "Missing required field: question"},
                status_code=400
            )

        result = process_question(question, files)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
