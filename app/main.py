from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .processor import process_question
import uvicorn

app = FastAPI()

@app.post("/api/")
async def analyze(request: Request):
    try:
        form = await request.form()

        # Read the files by their exact field names
        q_file = form.get("questions.txt")
        d_file = form.get("data.csv")

        if q_file is None or d_file is None:
            return JSONResponse(
                content={"error": "Both questions.txt and data.csv are required"},
                status_code=400
            )

        q_path = "uploaded_questions.txt"
        d_path = "uploaded_data.csv"

        with open(q_path, "wb") as f:
            f.write(await q_file.read())

        with open(d_path, "wb") as f:
            f.write(await d_file.read())

        result = process_question(d_path, q_path)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
