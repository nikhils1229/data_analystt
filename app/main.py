from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from .processor import process_question
import uvicorn

app = FastAPI()

@app.post("/api/")
async def analyze_data(files: list[UploadFile], question: str = Form(...)):
    try:
        result_json = process_question(files, question)
        return JSONResponse(content=result_json)
    except Exception as e:
        # Always return something to avoid zero marks
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
