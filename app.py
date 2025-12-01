import gradio as gr
import os
import time
from PIL import Image, ImageDraw
import numpy as np

# Import các module từ bộ lõi AI
from modules.detection import TextDetector
from modules.ocr import UniversalOCR
from modules.translator import LocalLLMTranslator
from modules.inpainting import Inpainter

# --- BIẾN TOÀN CỤC ĐỂ LƯU MODEL (TRÁNH RELOAD LẠI NHIỀU LẦN) ---
MODELS = {
    "detector": None,
    "ocr": None,
    "translator": None,
    "inpainter": None,
    "current_lang": None
}

def load_ai_models(lang_code):
    """Hàm khởi tạo model, chỉ chạy 1 lần hoặc khi đổi ngôn ngữ OCR"""
    global MODELS
    status_msg = ""
    
    # 1. Detection
    if MODELS["detector"] is None:
        status_msg += "⏳ Đang tải Text Detector...\n"
        MODELS["detector"] = TextDetector()
        
    # 2. OCR (Reload nếu đổi ngôn ngữ)
    ocr_lang_map = {'jp': 'japan', 'en': 'en', 'cn': 'ch', 'th': 'th', 'vi': 'vi'}
    target_ocr_lang = ocr_lang_map.get(lang_code, 'en')
    
    if MODELS["ocr"] is None or MODELS["current_lang"] != lang_code:
        status_msg += f"⏳ Đang tải OCR ({target_ocr_lang})...\n"
        # Xóa model cũ khỏi VRAM nếu cần thiết (ở đây tạm bỏ qua để đơn giản)
        MODELS["ocr"] = UniversalOCR(lang=target_ocr_lang)
        MODELS["current_lang"] = lang_code

    # 3. Translator (Nặng nhất)
    if MODELS["translator"] is None:
        status_msg += "⏳ Đang tải Qwen LLM (Translator)...\n"
        MODELS["translator"] = LocalLLMTranslator()

    # 4. Inpainter
    if MODELS["inpainter"] is None:
        status_msg += "⏳ Đang tải LaMa (Inpainter)...\n"
        MODELS["inpainter"] = Inpainter()
        
    return status_msg + "✅ Tất cả Model đã sẵn sàng!"

def process_manga(image_path, lang_code, progress=gr.Progress()):
    """Hàm xử lý chính gọi từ UI"""
    if image_path is None:
        return None, None, "Vui lòng upload ảnh!"

    # Load models nếu chưa có
    progress(0.1, desc="Kiểm tra Model...")
    load_log = load_ai_models(lang_code)
    
    try:
        # 1. Detect
        progress(0.3, desc="Đang tìm khung thoại...")
        detector = MODELS["detector"]
        bboxes = detector.detect(image_path)
        
        # Vẽ box lên ảnh gốc để preview
        original_img = Image.open(image_path).convert("RGB")
        preview_img = original_img.copy()
        draw = ImageDraw.Draw(preview_img)
        for box in bboxes:
            draw.rectangle(box, outline="red", width=3)
        
        if len(bboxes) == 0:
            return preview_img, original_img, "⚠️ Không tìm thấy khung thoại nào!"

        # 2. OCR
        progress(0.5, desc="Đang đọc chữ...")
        ocr_engine = MODELS["ocr"]
        raw_texts = []
        for box in bboxes:
            text = ocr_engine.run(original_img, box)
            if len(text.strip()) == 0: text = "..."
            raw_texts.append(text)

        # 3. Translate
        progress(0.7, desc="AI đang dịch...")
        translator = MODELS["translator"]
        translated_texts = translator.translate(raw_texts, source_lang=lang_code)

        # 4. Inpaint
        progress(0.9, desc="Đang xóa chữ...")
        inpainter = MODELS["inpainter"]
        clean_image = inpainter.remove_text(image_path, bboxes)

        # Format kết quả text
        result_text = ""
        for i, (raw, trans) in enumerate(zip(raw_texts, translated_texts)):
            result_text += f"[Box {i+1}]\nORIGIN: {raw}\nVIET: {trans}\n\n"

        return preview_img, clean_image, result_text

    except Exception as e:
        return None, None, f"❌ Lỗi xử lý: {str(e)}"

# --- GIAO DIỆN GRADIO ---
with gr.Blocks(title="AI Manga Translator Pro", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Manga Translator Pro")
    gr.Markdown("Công cụ dịch truyện tranh tự động sử dụng: YOLO (Detect) + PaddleOCR + Qwen-7B (Dịch) + LaMa (Xóa chữ).")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            input_img = gr.Image(type="filepath", label="Upload trang truyện", height=600)
            lang_dropdown = gr.Dropdown(
                choices=["jp", "en", "cn", "th", "vi"], 
                value="jp", 
                label="Ngôn ngữ gốc"
            )
            btn_run = gr.Button("🚀 DỊCH NGAY", variant="primary")
            
        with gr.Column(scale=2):
            # Output
            with gr.Tab("Kết quả hình ảnh"):
                with gr.Row():
                    out_detect = gr.Image(label="Phát hiện khung thoại", type="pil")
                    out_clean = gr.Image(label="Ảnh đã xóa chữ (Clean)", type="pil")
            
            with gr.Tab("Bản dịch Text"):
                out_text = gr.Textbox(label="Nội dung dịch (Song ngữ)", lines=20, show_copy_button=True)

    # Sự kiện click nút
    btn_run.click(
        fn=process_manga,
        inputs=[input_img, lang_dropdown],
        outputs=[out_detect, out_clean, out_text]
    )

if __name__ == "__main__":
    # share=True để tạo link public chạy trên Colab/Kaggle
    demo.launch(share=True, debug=True)
