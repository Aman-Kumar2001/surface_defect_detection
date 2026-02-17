import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
import time
import cv2
import json

MODEL_PATH = "models/surface_defect_model.keras"
CLASS_PATH = "models/class_names.json"
THRESHOLD_SCORE = 0.6

model = keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, 'r') as f:
    class_names = json.load(f)

def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224,224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_input = preprocess_input(frame_rgb)
    frame_input = np.expand_dims(frame_input, axis=0)

    return frame_input


cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise RuntimeError("Could not open webcam")


while True:
    ret, frame = cam.read()

    if not ret:
        break

    processed_frame = preprocess_frame(frame)

    pred = model.predict(processed_frame)
    idx = np.argmax(pred)
    score = float(np.max(pred))

    label_text = "Low Confidence"

    if score > THRESHOLD_SCORE :
        label_text = f"{class_names[idx]}, confidence:{score}"

    cv2.putText(frame, label_text, (20,80),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 1)

    cv2.imshow("Surface Defect Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows