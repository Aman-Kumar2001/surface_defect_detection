import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input


MODEL_PATH = "models/surface_defect_model.keras"
CLASS_PATH = "models/class_names.json"

model = keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, 'r') as f:
    class_names = json.load(f)


def process_image(image_bytes):
    
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)
    return image

def predict(image_bytes):
    image = process_image(image_bytes)

    pred = model.predict(image)
    idx = np.argmax(pred)
    score = float(np.max(pred))

    if score < 0.2:
        return {"message" : "Consider as no defect found",
                "Probable Defect" : class_names[idx],
                "Confidence score": score}
    elif score < 0.4:
        return {"Defect" : class_names[idx],
                "Confidence score": score,
                "message": "Low Cofidence"}
    
    return {"Defect found" : class_names[idx],
            "Confidence score": score}