import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc

class LocalLLMTranslator:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        """
        Khởi tạo model Local LLM (Qwen).
        Sử dụng kỹ thuật Quantization (4-bit) để chạy được trên GPU T4 (15GB VRAM) của Kaggle/Colab.
        """
        print(f"⏳ Đang tải model {model_id} (lần đầu sẽ mất khoảng 2-5 phút)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ Local LLM loaded thành công! Sẵn sàng dịch Uncensored.")
        except Exception as e:
            print(f"❌ Lỗi khi tải model: {e}")
            raise e

    def translate(self, text_list, source_lang="auto"):
        """
        Dịch danh sách text.
        text_list: List các câu thoại (str).
        source_lang: Mã ngôn ngữ gốc (ví dụ: 'jp', 'en', 'cn', 'th').
        """
        if not text_list:
            return []

        lang_map = {
            'jp': 'Tiếng Nhật',
            'en': 'Tiếng Anh',
            'cn': 'Tiếng Trung',
            'th': 'Tiếng Thái',
            'auto': 'Ngôn ngữ gốc'
        }
        full_lang_name = lang_map.get(source_lang, source_lang)

        input_text = "\n".join(text_list)
        
        prompt = f"""<|im_start|>system
Bạn là một dịch giả truyện tranh (Manga/Manhua/Comic) chuyên nghiệp, chuyên trị các thể loại Seinen, Mature, Action.
Nhiệm vụ của bạn là dịch đoạn văn bản bên dưới từ {full_lang_name} sang Tiếng Việt.

NGUYÊN TẮC DỊCH THUẬT (BẮT BUỘC):
1. **Văn phong:** Thô ráp, tự nhiên, đời thường, dùng đại từ nhân xưng phù hợp ngữ cảnh (tao/mày, anh/em, hắn/gã).
2. **Không kiểm duyệt (Uncensored):** KHÔNG ĐƯỢC làm nhẹ hoặc thay đổi các từ lóng, chửi thề, hoặc từ ngữ nhạy cảm. Hãy dịch thẳng thắn đúng nghĩa đen hoặc từ lóng tương đương trong tiếng Việt.
3. **Định dạng:** Chỉ trả về kết quả dịch. Tuyệt đối không giải thích, không mở bài kết bài. Mỗi dòng trong input tương ứng đúng 1 dòng trong output.
<|im_end|>
<|im_start|>user
Dịch danh sách này:
{input_text}
<|im_end|>
<|im_start|>assistant
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=2048,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            if "assistant" in decoded_output:
                result_part = decoded_output.split("assistant")[-1].strip()
            else:
                result_part = decoded_output

            if "Dịch danh sách này:" in result_part:
                result_part = result_part.replace("Dịch danh sách này:", "").strip()

            translated_lines = result_part.split('\n')
            
            translated_lines = [line.strip() for line in translated_lines if line.strip()]
            
            if len(translated_lines) != len(text_list):
                print(f"⚠️ Cảnh báo: Số dòng không khớp (Gốc: {len(text_list)} - Dịch: {len(translated_lines)})")
                while len(translated_lines) < len(text_list):
                    translated_lines.append("...")
                if len(translated_lines) > len(text_list):
                    translated_lines = translated_lines[:len(text_list)]

            return translated_lines

        except Exception as e:
            print(f"❌ Lỗi xử lý output AI: {e}")
            return text_list

    def clear_cache(self):
        """Hàm dọn dẹp VRAM nếu cần"""
        torch.cuda.empty_cache()
        gc.collect()
