
# modules/detection.py
from comic_text_detector.inference import TextDetector as ComicTextDetector
import numpy as np
from PIL import Image

class TextDetector:
    def __init__(self, model_path=None):
        self.model = ComicTextDetector(model_path=model_path, input_size=1024, device='cuda')
        print("✅ Text Detector loaded!")

    def detect(self, image_path):
        """
        Trả về list các bounding box: [[x_min, y_min, x_max, y_max], ...]
        """
        detection_results = self.model(image_path)
        
        bboxes = detection_results[1] 
        
        final_bboxes = []
        for box in bboxes:
             x, y, w, h = box
             final_bboxes.append([int(x), int(y), int(x+w), int(y+h)])
             
        return final_bboxes
