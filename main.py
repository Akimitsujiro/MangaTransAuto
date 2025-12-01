import argparse
from modules.ocr import UniversalOCR 
from modules.translator import LocalLLMTranslator

def main(image_path, lang_code):
    print(f"🚀 Xử lý: {image_path} | Ngôn ngữ: {lang_code}")
    
    paddle_lang_map = {
        'jp': 'japan',
        'en': 'en',
        'cn': 'ch',
        'th': 'th'
    }
    
    ocr_lang = paddle_lang_map.get(lang_code, 'en')
    
    detector = TextDetector()
    ocr_engine = UniversalOCR(lang=ocr_lang)
    translator = LocalLLMTranslator()
    inpainter = Inpainter()
    
    
    raw_texts = []
    for box in bboxes:
        text = ocr_engine.run(original_img, box)
        raw_texts.append(text)
        
    print(f"📝 Gốc ({lang_code}):", raw_texts)
    
    translated_texts = translator.translate(raw_texts, source_lang=lang_code)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--lang", type=str, default="jp", choices=['jp', 'en', 'cn', 'th'], help="Ngôn ngữ gốc của truyện")
    args = parser.parse_args()
    
    main(args.img, args.lang)
