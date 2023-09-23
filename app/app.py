import io
import os

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse
import pandas as pd
import uvicorn
import tempfile
import nltk

from ml.model import Model

app = FastAPI()
model = None

@app.on_event("startup")
def startup_event():
    global model
    model = Model()


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    Perform a Health Check
    """
    return HealthCheck(status="OK")


@app.post("/process-csv/")
async def process_csv(file: UploadFile):
    """Read file and process data"""
    if file.filename.endswith(".csv"):
        csv_data = await file.read()
        df = pd.read_csv(io.BytesIO(csv_data))
        if not model:
            raise HTTPException(status_code=500,
                                detail="Sorry, we have problems with ML model. Please try few second later.")

        result = model.process(df)
        if 'channel_id' in df.columns:
            ds = df[['text', 'channel_id']].drop_duplicates(subset=['text'])
            result = result.merge(ds, on='text', how='left')
            result = result[['text', 'channel_id', 'category']]

        stream = io.StringIO()
        result.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"
        return response
    else:
        return {"message": "Invalid file format. Please upload an CSV file."}


def main() -> None:
    """Run application"""
    nltk.download('punkt')
    uvicorn.run("app:app", host="localhost")

if __name__ == "__main__":
    main()