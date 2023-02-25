import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from torchvision import models, transforms
import torch.nn as nn
import torch
from PIL import Image
from io import BytesIO
from inference import Inference
import traceback

device = "cpu"
class_names=['akorda', 'baiterek', 'khanshatyr', 'mangilikel', 'mosque', 'nuralem', 'piramida', 'shabyt']
num_classes = len(class_names)



app = FastAPI()

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    response = {}
    try:
        extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"
        image = await file.read()

        infer = Inference()
        response = infer.classify_image(image)
    except Exception as e:
        response["error"] = str(e)
        response["error_traceback"] = traceback.format_exc()

    return response