from transformers import AutoModelForCausalLM, AutoTokenizer


class ASLtoISLTranslator:
    def __init__(self, model_id="meta-llama/Llama-2-7b-chat-hf", quantize=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=quantize,
            device_map="auto"
        )

    def translate(self, asl_gloss):
        prompt = f"Convert the following ASL gloss to ISL gloss:\nASL: {asl_gloss}\nISL:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_length=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
