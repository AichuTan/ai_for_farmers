from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model ONCE
model = YOLO("model/best.pt")

def detect_disease(image: Image.Image):
    image_np = np.array(image.convert("RGB"))
    results = model(image_np)
    boxes = results[0].boxes

    if boxes and len(boxes.cls) > 0:
        class_id = int(boxes.cls[0])
        conf = float(boxes.conf[0]) * 100
        disease = results[0].names[class_id]
    else:
        class_id = None
        conf = 0
        disease = "No disease detected"

    annotated_img = results[0].plot()
    return {
        "disease": disease,
        "confidence": round(conf, 2),
        "annotated_image": annotated_img
    }

def get_all_disease_classes():
    return list(model.names.values())
