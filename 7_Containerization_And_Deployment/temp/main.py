from fastapi import FastAPI,Request, Body
import uvicorn

from models.schemas import image_predRequestModel
from contextlib import asynccontextmanager
from model_work import CatAndDogModel


ml_models = {}
@asynccontextmanager
async def lifespan(app: FastAPI):

    catAndDogModel = CatAndDogModel()
    catAndDogModel.load_model()

    ml_models["catAndDogModel"] = catAndDogModel


    yield
    ml_models.clear()








app = FastAPI(lifespan=lifespan)



@app.get("/")
def home():
    return {"message": "Hello, World!"}


@app.post("/predict")
def predict(request: Request, body: image_predRequestModel = Body(...)) -> image_predRequestModel:
    

    request_data = body
    catAndDogModel = ml_models["catAndDogModel"]

    img_array = catAndDogModel.preprocess_image(request_data.image)
    class_name, prediction = catAndDogModel.predict(img_array)
    str_output = f"Prediction: {class_name}, Confidence: {prediction:.2f}"
    request_data.class_name = str_output
    
    
    return request_data





if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8888)