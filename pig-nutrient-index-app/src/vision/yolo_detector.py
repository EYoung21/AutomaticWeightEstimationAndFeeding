import torch
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov5s.pt', device='cpu'):
        # Load the trained YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.device = device

    def detect_pigs(self, image_path):
        # Read image
        img = cv2.imread(image_path)
        # Run YOLO inference
        results = self.model(img)
        # Extract bounding boxes for pigs (class 0 if only one class)
        pig_boxes = []
        for *box, conf, cls in results.xyxy[0].tolist():
            if int(cls) == 0:  # Assuming 'pig' is class 0
                pig_boxes.append(box)
        return pig_boxes