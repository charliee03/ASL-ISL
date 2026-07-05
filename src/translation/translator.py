import json
import re
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


class ASLtoISLTranslator:
    def __init__(self, model_id="meta-llama/Llama-2-7b-chat-hf", quantize=True,
                 vocabulary_path="configs/isl_vocabulary.json"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=quantize,
            device_map="auto"
        )
        self.vocabulary_path = Path(vocabulary_path)
        self.vocabulary = self._load_vocabulary()

    def _load_vocabulary(self):
        if not self.vocabulary_path.exists():
            return set()

        with open(self.vocabulary_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        entries = payload.get("entries", [])
        return {
            entry.get("gloss", "").strip().lower()
            for entry in entries
            if entry.get("gloss")
        }

    def validate_output(self, isl_gloss):
        tokens = [token for token in re.split(r"\W+", isl_gloss.lower()) if token]
        valid_tokens = [token for token in tokens if token in self.vocabulary]
        if valid_tokens:
            return " ".join(valid_tokens)
        return isl_gloss.strip()

    def translate(self, asl_gloss):
        prompt = f"Convert the following ASL gloss to ISL gloss:\nASL: {asl_gloss}\nISL:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_length=128)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        isl_gloss = generated_text.split("ISL:", 1)[-1].strip()
        return self.validate_output(isl_gloss)
