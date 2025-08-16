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
    This version is specifically adapted to handle the evaluation format where
    the question and data files are sent in multipart/form-data fields
    named after the original filenames (e.g., 'questions.txt', 'data.csv').
    """
    question = None
    files = []
    
    content_type = request.headers.get("content-type", "")

    try:
        # This block handles JSON requests, which might still be needed for other tests.
        if "application/json" in content_type:
            data = await request.json()
            question = data.get("question") or (data.get("vars", {})).get("question")
            if "files" in data and isinstance(data["files"], list):
                for f in data["files"]:
                    if "filename" in f and "content" in f:
                        decoded_content = base64.b64decode(f["content"])
                        file_obj = io.BytesIO(decoded_content)
                        files.append(UploadFile(filename=f["filename"], file=file_obj))

        # This block is updated for the specific multipart/form-data evaluation format.
        elif "multipart/form-data" in content_type:
            form_data = await request.form()
            for name, file_upload in form_data.items():
                # The question is in a field named after its file, e.g., 'questions.txt'
                if name.endswith('.txt'):
                    # Read the content of the UploadFile to get the question string
                    question_bytes = await file_upload.read()
                    question = question_bytes.decode('utf-8')
                else:
                    # Treat all other files as data files to be processed
                    files.append(file_upload)
        
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
        return JSONResponse(content={"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
