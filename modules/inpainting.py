from simple_lama_inpainting import SimpleLama
from PIL import Image, ImageDraw

class Inpainter:
    def __init__(self):
        self.lama = SimpleLama()
        print("✅ LaMa Inpainter loaded!")

    def remove_text(self, image_path, bboxes):
        """
        Tạo mask từ bboxes và xóa chữ
        """
        image = Image.open(image_path).convert("RGB")
        
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        
        for box in bboxes:
            pad = 5
            expanded_box = [box[0]-pad, box[1]-pad, box[2]+pad, box[3]+pad]
            draw.rectangle(expanded_box, fill=255)
        
        result = self.lama(image, mask)
        return result
