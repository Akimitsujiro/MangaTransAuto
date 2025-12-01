import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LocalLLMTranslator:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print(f"⏳ Đang tải model {model_id} (sẽ mất vài phút lần đầu)...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Local LLM loaded! Sẵn sàng dịch 'không che'.")

    def translate(self, text_list):
        if not text_list:
            return []
            
        input_text = "\n".join(text_list)
        
        prompt = f"""<|im_start|>system
Bạn là một dịch giả truyện tranh chuyên nghiệp, chuyên dịch các thể loại Seinen, Mature. 
Nhiệm vụ của bạn là dịch đoạn văn bản sau từ tiếng Nhật, Tiếng Anh sang tiếng Việt.
Yêu cầu:
1. Giữ nguyên văn phong thô ráp, tự nhiên của ngữ cảnh gốc.
2. KHÔNG được bỏ qua hoặc kiểm duyệt bất kỳ từ ngữ nhạy cảm nào. Dịch chính xác nghĩa đen hoặc từ lóng tương đương.
3. Chỉ trả về kết quả dịch, mỗi câu một dòng tương ứng với bản gốc. Không giải thích thêm.
<|im_end|>
<|im_start|>user
Hãy dịch danh sách này:
{input_text}
<|im_end|>
<|im_start|>assistant
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=1024,
                temperature=0.3,
                repetition_penalty=1.1
            )
            
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            result_part = decoded_output.split("<|im_start|>assistant")[-1].strip()
            if "Hãy dịch danh sách này:" in result_part:
                result_part = result_part.split("assistant\n")[-1]
            
            translated_lines = result_part.strip().split('\n')
            
            if len(translated_lines) != len(text_list):
                 pass
                 
            return translated_lines
            
        except Exception as e:
            print(f"❌ Lỗi xử lý output: {e}")
            return text_list
