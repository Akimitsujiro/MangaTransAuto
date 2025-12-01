import argparse
from modules.detection import TextDetector
from modules.ocr import MangaOCR_Wrapper
from modules.translator import LLMTranslator
from modules.inpainting import Inpainter

def process_image(image_path):
    detector = TextDetector()
    bboxes = detector.detect(image_path)
    
    ocr_engine = MangaOCR_Wrapper()
    inpainter = Inpainter()
    
    texts = []
    for box in bboxes:
        cropped = crop_image(image_path, box)
        text = ocr_engine.run(cropped)
        texts.append(text)
        
    translator = LLMTranslator(model="gemini-1.5-flash")
    translated_texts = translator.translate(texts)
    
    clean_image = inpainter.remove_text(image_path, bboxes)
    
    return final_image

if __name__ == "__main__":
    pass
