from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

class UniversalOCR:
    def __init__(self, lang='en'):
        """
        lang support: 
        'ch' (Trung), 'en' (Anh), 'japan' (Nhật), 'korean' (Hàn)
        Lưu ý: Thái Lan thì dùng lang='structure' hoặc model riêng, 
        nhưng PaddleOCR mặc định hỗ trợ đa ngữ khá tốt. 
        Với tiếng Thái cần cài đặt riêng: lang='th' (cần check support model)
        """
        print(f"⏳ Đang tải PaddleOCR cho ngôn ngữ: {lang}...")
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False) 
        print("✅ PaddleOCR loaded!")

    def run(self, image, bbox):
        """
        image: PIL Image
        bbox: [x_min, y_min, x_max, y_max]
        """
        cropped = image.crop(bbox)
        
        img_np = np.array(cropped)
        
        result = self.ocr.ocr(img_np, cls=True)
        
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                text_content = line[1][0]
                text_lines.append(text_content)

        return " ".join(text_lines)
