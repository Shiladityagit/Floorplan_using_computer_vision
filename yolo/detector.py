from ultralytics import YOLO
import cv2

def run_detection(image_path, model_path):
    model = YOLO(model_path)
    results = model(image_path)
    class_names = model.names
    original_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return results, class_names, original_img
