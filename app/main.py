# main.py

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from typing import List
from .processor import process_question
import uvicorn
import base64
import io

app = FastAPI()

@app.post("/api/")
async def analyze(request: Request):
    """
    Analyzes a user's question about provided data files.

    This endpoint can handle both `multipart/form-data` and `application/json`
    requests. It extracts the question and any associated files, then
    delegates processing to the `process_question` function.
    """
    question = None
    files = []
    
    # Check the content type to handle JSON or form-data
    content_type = request.headers.get("content-type", "")

    try:
        if "application/json" in content_type:
            # Handle JSON payload
            data = await request.json()
            question = data.get("question") or (data.get("vars", {})).get("question")
            
            # Handle base64 encoded files embedded in JSON
            if "files" in data and isinstance(data["files"], list):
                for f in data["files"]:
                    if "filename" in f and "content" in f:
                        decoded_content = base64.b64decode(f["content"])
                        file_obj = io.BytesIO(decoded_content)
                        files.append(UploadFile(filename=f["filename"], file=file_obj))

        elif "multipart/form-data" in content_type:
            # Handle multipart form data
            form_data = await request.form()
            question = form_data.get("question")
            # The 'files' key in a form can have multiple parts
            form_files = form_data.getlist("files")
            if form_files:
                files.extend(form_files)
        
        # Fallback for question in query parameters
        if not question:
            question = request.query_params.get("question")

        if not question:
            return JSONResponse(
                content={"error": "Missing required field: question"},
                status_code=400
            )

        # Call the processor with the extracted data
        result = process_question(question, files)
        return JSONResponse(content=result)

    except Exception as e:
        # Generic error handler for unexpected issues
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
