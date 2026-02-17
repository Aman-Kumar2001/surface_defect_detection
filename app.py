from fastapi import FastAPI, File, UploadFile , HTTPException
from fastapi.responses import JSONResponse
from preprocessing.preprocess_image import predict


app = FastAPI()

@app.get("/")
def home():
    return {"message" : "Server is running properly"}

@app.get("/health")
def health():
    return {"Status": "OK"}

@app.post("/predict")
async def predict_image(file : UploadFile = File(...)):

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise ValueError("Empty file uploaded")

        result = predict(image_bytes)

        return JSONResponse(content=result, status_code=200)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

