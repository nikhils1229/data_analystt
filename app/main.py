from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import .processor

app = FastAPI()

@app.post("/api/")
async def analyze(
    questions: UploadFile = File(..., alias="questions.txt"),   # required
    files: Optional[List[UploadFile]] = File(None)              # optional
):
    try:
        # Read questions
        questions_text = (await questions.read()).decode("utf-8")

        # Read any uploaded files (CSV, etc.)
        file_contents = {}
        if files:
            for f in files:
                file_contents[f.filename] = await f.read()

        # Call processor
        result = processor.process(questions_text, file_contents)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
