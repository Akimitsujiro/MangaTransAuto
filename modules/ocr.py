from manga_ocr import MangaOcr
from PIL import Image

class MangaOCR_Wrapper:
    def __init__(self):
        self.ocr = MangaOcr()
        print("✅ Manga OCR loaded!")

    def run(self, image, bbox):
        """
        image: PIL Image object gốc
        bbox: [x_min, y_min, x_max, y_max]
        """
        cropped_image = image.crop(bbox)
        
        text = self.ocr(cropped_image)
        return text
