from fastapi import FastAPI, UploadFile, Form, File, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
from .processor import process_question
import uvicorn
import base64
import io

app = FastAPI()

@app.post("/api/")
async def analyze(
    request: Request,
    question: Optional[str] = Form(None),
    files: List[UploadFile] = File(default=[])
):
    try:
        # If JSON request
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            
            # Extract question from different possible JSON structures
            if not question:
                if isinstance(data, dict):
                    question = data.get("question") or data.get("vars", {}).get("question")

            # Extract base64 CSV files from JSON
            if "files" in data and isinstance(data["files"], list):
                json_files = []
                for f in data["files"]:
                    if "filename" in f and "content" in f:
                        try:
                            decoded = base64.b64decode(f["content"])
                            file_obj = io.BytesIO(decoded)
                            upload = UploadFile(filename=f["filename"], file=file_obj)
                            json_files.append(upload)
                        except Exception as e:
                            print(f"Error decoding file {f.get('filename')}: {e}")
                files = json_files

        # Fallback: query param
        if not question:
            question = request.query_params.get("question")

        if not question:
            return JSONResponse(content={"error": "Missing required field: question"}, status_code=400)

        # Call processing function
        result = process_question(question, files)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
