from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from processor import process_question
import uvicorn

app = FastAPI()

@app.post("/api/")
async def analyze(
    questions_txt: UploadFile = File(..., alias="questions.txt"),
    data_csv: UploadFile = File(..., alias="data.csv"),
):
    try:
        # Save uploaded files to temp paths
        q_path = "uploaded_questions.txt"
        d_path = "uploaded_data.csv"

        with open(q_path, "wb") as f:
            f.write(await questions_txt.read())

        with open(d_path, "wb") as f:
            f.write(await data_csv.read())

        # Call processor
        result = process_question(d_path, q_path)
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
