# main.py

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import base64
import io
from .processor import process_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    files: Optional[List[dict]] = None

@app.post("/api/")
async def analyze(request: Request):
    """
    Analyzes a user's question about provided data files.
    This endpoint handles both JSON requests and form-data requests.
    """
    question = None
    files = []

    try:
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            # Handle JSON payload
            data = await request.json()
            question = data.get("question")

            # Also check nested structure that might be used by test framework
            if not question:
                vars_data = data.get("vars", {})
                question = vars_data.get("question")

            # Handle base64 encoded files in JSON
            if "files" in data and isinstance(data["files"], list):
                for f in data["files"]:
                    if "filename" in f and "content" in f:
                        try:
                            decoded_content = base64.b64decode(f["content"])
                            file_obj = io.BytesIO(decoded_content)
                            # Create a simple file-like object
                            class SimpleFile:
                                def __init__(self, filename, file_obj):
                                    self.filename = filename
                                    self.file = file_obj
                            files.append(SimpleFile(f["filename"], file_obj))
                        except Exception as e:
                            print(f"Error decoding file {f.get('filename', 'unknown')}: {e}")

        elif "multipart/form-data" in content_type:
            # Handle multipart form data
            form_data = await request.form()
            question = form_data.get("question")

            # Handle uploaded files
            form_files = form_data.getlist("files")
            if form_files:
                for f in form_files:
                    if hasattr(f, 'filename') and hasattr(f, 'file'):
                        files.append(f)

        # Also check query parameters as fallback
        if not question:
            question = request.query_params.get("question")

        # Handle the case where question might be empty string from request body
        if not question or question.strip() == "":
            return JSONResponse(
                content={"error": "Missing required field: question"}, 
                status_code=400
            )

        # Process the question
        result = process_question(question, files)

        # Ensure result is JSON serializable
        if isinstance(result, dict):
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"error": "Invalid response format from processor"})

    except Exception as e:
        print(f"API Error: {str(e)}")
        return JSONResponse(
            content={"error": f"An unexpected error occurred: {str(e)}"}, 
            status_code=500
        )

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
