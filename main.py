import argparse
import os
from PIL import Image
from modules.detection import TextDetector
from modules.ocr import MangaOCR_Wrapper
from modules.translator import LLMTranslator
from modules.inpainting import Inpainter

def main(image_path):
    print(f"🚀 Đang xử lý: {image_path}")
    
    detector = TextDetector()
    ocr_engine = MangaOCR_Wrapper()
    translator = LLMTranslator()
    inpainter = Inpainter()
    
    bboxes = detector.detect(image_path)
    print(f"🔍 Tìm thấy {len(bboxes)} khung thoại.")
    
    raw_texts = []
    original_img = Image.open(image_path)
    
    for box in bboxes:
        text = ocr_engine.run(original_img, box)
        raw_texts.append(text)
        
    print("🇯🇵 Gốc:", raw_texts)
    
    print("AI đang dịch...")
    translated_texts = translator.translate(raw_texts)
    print("🇻🇳 Việt:", translated_texts)
    
    print("AI đang xóa chữ...")
    clean_image = inpainter.remove_text(image_path, bboxes)
    
    clean_path = "output_clean.jpg"
    clean_image.save(clean_path)
    print(f"✨ Đã lưu ảnh sạch tại: {clean_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Đường dẫn ảnh manga")
    args = parser.parse_args()
    
    main(args.img)
