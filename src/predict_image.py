import numpy as np
import cv2
import json
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
import sys

MODEL_PATH = "models/surface_defect_model.keras"
CLASS_PATH = "models/class_names.json"

model = keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, 'r') as f:
    class_names = json.load(f)


def process_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))
    image = preprocess_input(image)

    image_input = np.expand_dims(image, axis=0)

    return image_input


def predict(image_path):
    image = process_image(image_path)

    pred = model.predict(image)
    idx = np.argmax(pred)
    score = float(np.max(pred))

    return class_names[idx], score

if __name__ == "__main__":
    image_path = sys.argv[1]
    label, score = predict(image_path)

    print(f'This defect comes under {label}')
    print(f'The confidence score is:  {score}')