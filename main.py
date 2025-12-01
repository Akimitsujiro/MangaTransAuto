import argparse
import os
import time
from PIL import Image, ImageDraw, ImageFont

# Import các module (đảm bảo bạn đã tạo folder modules và có các file .py bên trong)
from modules.detection import TextDetector
from modules.ocr import UniversalOCR
from modules.translator import LocalLLMTranslator
from modules.inpainting import Inpainter

def save_result_text(image_path, translations):
    """Lưu kết quả dịch ra file text để tiện xem"""
    base_name = os.path.splitext(image_path)[0]
    txt_path = f"{base_name}_translated.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(translations))
    return txt_path

def main(image_path, lang_code, use_gpu=True):
    start_time = time.time()
    print(f"\n🚀 BẮT ĐẦU XỬ LÝ: {image_path}")
    print(f"🌍 Ngôn ngữ gốc: {lang_code.upper()}")

    # --- BƯỚC 1: KHỞI TẠO CÁC MODEL ---
    print("\n[1/5] ⏳ Đang khởi tạo các AI Models...")
    
    # Mapping ngôn ngữ cho OCR
    ocr_lang_map = {
        'jp': 'japan', 'en': 'en', 
        'cn': 'ch', 'th': 'th', 'vi': 'vi'
    }
    
    try:
        detector = TextDetector() # Tự tải model detection
        ocr_engine = UniversalOCR(lang=ocr_lang_map.get(lang_code, 'en'))
        translator = LocalLLMTranslator() # Load Qwen (nặng nhất)
        inpainter = Inpainter() # Load LaMa
    except Exception as e:
        print(f"❌ Lỗi khởi tạo model: {e}")
        return

    # --- BƯỚC 2: PHÁT HIỆN KHUNG THOẠI ---
    print("\n[2/5] 🔍 Đang tìm khung thoại (Detection)...")
    bboxes = detector.detect(image_path)
    print(f"   👉 Tìm thấy {len(bboxes)} khung thoại.")

    if len(bboxes) == 0:
        print("⚠️ Không tìm thấy khung thoại nào. Dừng xử lý.")
        return

    # --- BƯỚC 3: ĐỌC CHỮ (OCR) ---
    print("\n[3/5] 📖 Đang đọc chữ (OCR)...")
    original_img = Image.open(image_path).convert("RGB")
    raw_texts = []
    
    for i, box in enumerate(bboxes):
        text = ocr_engine.run(original_img, box)
        # Lọc bớt text rác quá ngắn
        if len(text.strip()) == 0: 
            text = "..."
        raw_texts.append(text)
        print(f"   Box {i+1}: {text}")

    # --- BƯỚC 4: DỊCH THUẬT (TRANSLATION) ---
    print("\n[4/5] 🧠 AI đang dịch (Translation)...")
    translated_texts = translator.translate(raw_texts, source_lang=lang_code)
    
    # In kết quả so sánh
    print("-" * 30)
    for i, (raw, trans) in enumerate(zip(raw_texts, translated_texts)):
        print(f"🔸 {raw}")
        print(f"🔹 {trans}")
        print("-" * 10)

    # Lưu file text kết quả
    txt_file = save_result_text(image_path, translated_texts)
    print(f"✅ Đã lưu bản dịch text tại: {txt_file}")

    # --- BƯỚC 5: XÓA CHỮ (INPAINTING) ---
    print("\n[5/5] 🎨 Đang xóa chữ gốc (Inpainting)...")
    clean_image = inpainter.remove_text(image_path, bboxes)
    
    # Lưu ảnh sạch
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    clean_path = f"output_{base_name}_cleaned.png"
    clean_image.save(clean_path)
    print(f"✨ Đã lưu ảnh sạch tại: {clean_path}")

    total_time = time.time() - start_time
    print(f"\n🎉 HOÀN TẤT! Tổng thời gian: {total_time:.2f} giây.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Đường dẫn đến file ảnh truyện")
    parser.add_argument("--lang", type=str, default="jp", choices=['jp', 'en', 'cn', 'th', 'vi'], help="Ngôn ngữ gốc (jp, en, cn, th)")
    args = parser.parse_args()
    
    if not os.path.exists(args.img):
        print(f"❌ Lỗi: Không tìm thấy file ảnh tại {args.img}")
    else:
        main(args.img, args.lang)
