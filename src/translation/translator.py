"""Rule-based ASL-gloss to draft ISL-gloss conversion with optional LLM refinement."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)


class ASLtoISLTranslator:
    """Translate gloss tokens while keeping the offline rule baseline available."""

    def __init__(
        self,
        gloss_vocab_path: Optional[str] = None,
        grammar_rules_path: Optional[str] = None,
        config_path: Optional[str] = None,
        model_id: str = "meta-llama/Llama-2-7b-chat-hf",
        quantize: bool = True,
        device: Optional[str] = None,
        enable_llm: bool = False,
    ):
        self.gloss_vocab_path = gloss_vocab_path or "models/recognition/gloss_vocab.json"
        self.grammar_rules_path = grammar_rules_path or "configs/grammar_rules.json"
        self.config_path = config_path or "configs/translation.yaml"
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gloss_vocab = self._load_gloss_vocab()
        self.grammar_rules = self._load_grammar_rules()
        self.config = self._load_config()
        self.max_length = self.config.get("model", {}).get("max_length", 128)
        self.temperature = self.config.get("model", {}).get("temperature", 0.7)
        self.top_p = self.config.get("model", {}).get("top_p", 0.9)
        self.tokenizer = None
        self.model = None
        if enable_llm:
            self._load_llm(quantize)
        else:
            logger.info("LLM loading disabled; using deterministic draft rules")

    def _load_gloss_vocab(self) -> Dict[str, int]:
        path = Path(self.gloss_vocab_path)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("gloss_to_id"), dict):
                return data["gloss_to_id"]
            if isinstance(data.get("id_to_gloss"), dict):
                return {gloss: int(index) for index, gloss in data["id_to_gloss"].items()}
            if data and isinstance(next(iter(data.values())), int):
                return data
            return {gloss: index for index, gloss in data.items()}
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
            logger.exception("Could not load gloss vocabulary from %s", path)
            return {}

    def _load_grammar_rules(self) -> dict:
        path = Path(self.grammar_rules_path)
        if not path.is_file():
            return {"gloss_mappings": {}, "filter_rules": {}, "special_cases": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not load grammar rules from %s", path)
            return {"gloss_mappings": {}, "filter_rules": {}, "special_cases": {}}

    def _load_config(self) -> dict:
        path = Path(self.config_path)
        if not path.is_file():
            return {}
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            logger.exception("Could not load translation config from %s", path)
            return {}

    def _load_llm(self, quantize: bool) -> None:
        try:
            configured_id = self.config.get("model", {}).get("model_id", self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(configured_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            kwargs = {"device_map": "auto"}
            if quantize and torch.cuda.is_available():
                kwargs["load_in_4bit"] = True
            self.model = AutoModelForCausalLM.from_pretrained(configured_id, **kwargs)
            self.model.eval()
        except Exception as error:  # Optional network/model/runtime path.
            logger.warning("Optional LLM unavailable; using rules: %s", error)
            self.tokenizer = None
            self.model = None

    def _filter_glosses(self, glosses: List[str]) -> List[str]:
        fillers = self.grammar_rules.get("filter_rules", {}).get("filler_words", [])
        filler_set = {word.upper() for word in fillers}
        return [gloss for gloss in glosses if gloss.upper() not in filler_set]

    def _apply_grammar_rules(self, glosses: List[str]) -> List[str]:
        mappings = self.grammar_rules.get("gloss_mappings", {})
        special = self.grammar_rules.get("special_cases", {})
        pronouns = special.get("pronouns", {})
        question_markers = special.get("question_markers", {})
        result = []
        for gloss in glosses:
            key = gloss.upper()
            result.append(mappings.get(key, pronouns.get(key, question_markers.get(key, gloss))))
        return result

    @staticmethod
    def _parse_llm_output(output: str) -> List[str]:
        line = output.strip().splitlines()[-1] if output.strip() else ""
        return [token.strip(".,!?;:\"'") for token in line.split() if token.strip(".,!?;:\"'")]

    def translate(self, asl_glosses: List[str]) -> List[str]:
        if not asl_glosses:
            return []
        filtered = self._filter_glosses(asl_glosses)
        rule_based = self._apply_grammar_rules(filtered)
        if self.model is None or self.tokenizer is None:
            return rule_based
        prompt = (
            "Convert this ASL gloss sequence to ISL glosses. Return glosses only.\n"
            f"ASL: {' '.join(filtered)}\nISL:"
        )
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    do_sample=self.temperature > 0,
                    temperature=max(self.temperature, 1e-5),
                    top_p=self.top_p,
                )
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated = decoded.rsplit("ISL:", 1)[-1]
            parsed = self._parse_llm_output(translated)
            return parsed or rule_based
        except Exception as error:
            logger.warning("LLM translation failed; using rules: %s", error)
            return rule_based

    def translate_batch(self, batch_glosses: List[List[str]]) -> List[List[str]]:
        return [self.translate(glosses) for glosses in batch_glosses]

    def translate_gloss_string(self, asl_gloss_string: str) -> str:
        return " ".join(self.translate(asl_gloss_string.strip().split()))
