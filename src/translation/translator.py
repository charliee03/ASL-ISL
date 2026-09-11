"""Rule-based ASL-gloss to draft ISL-gloss conversion with optional LLM refinement."""

import json
import logging
import os
import string
import csv
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
        enable_gemini: bool = False,
    ):
        self.gloss_vocab_path = gloss_vocab_path or "models/recognition/gloss_vocab.json"
        self.grammar_rules_path = grammar_rules_path or "configs/grammar_rules.json"
        self.config_path = config_path or "configs/translation.yaml"
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gloss_vocab = self._load_gloss_vocab()
        self.grammar_rules = self._load_grammar_rules()
        self.curated_sentence_overrides = self._load_curated_sentence_overrides()
        self.config = self._load_config()
        self.max_length = self.config.get("model", {}).get("max_length", 128)
        self.temperature = self.config.get("model", {}).get("temperature", 0.7)
        self.top_p = self.config.get("model", {}).get("top_p", 0.9)
        self.tokenizer = None
        self.model = None
        self.gemini_client = None
        self.gemini_model = os.getenv("AITE_GEMINI_MODEL", "gemini-2.5-flash")
        self.gemini_allowed_glosses = self._load_gemini_allowed_glosses()
        if enable_llm:
            self._load_llm(quantize)
        if enable_gemini:
            self._load_gemini()
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

    def _load_curated_sentence_overrides(self) -> Dict[str, List[str]]:
        """Load approved exact-sentence drafts from the local review sheet."""
        overrides = {
            str(source).casefold(): [str(token) for token in target if str(token).strip()]
            for source, target in self.grammar_rules.get("sentence_overrides", {}).items()
            if isinstance(target, list)
        }
        review_location = self.grammar_rules.get("curated_sentence_review_csv")
        if not isinstance(review_location, str) or not review_location.strip():
            return overrides
        review_path = Path(self.grammar_rules_path).parent.parent / review_location
        try:
            with review_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("review_status", "").strip().casefold() != "approved":
                        continue
                    source = row.get("source_sentence", "").strip()
                    draft = row.get("draft_hinglish_text", "").strip().strip(string.punctuation)
                    if source and draft:
                        # Explicit configuration entries are the higher-priority
                        # corrections when they disagree with the review draft.
                        overrides.setdefault(source.casefold(), draft.split())
        except OSError as error:
            logger.warning("Could not load curated sentence review sheet %s: %s", review_path, error)
        return overrides

    def _load_gemini_allowed_glosses(self) -> set[str]:
        """Load the local glosses that the renderer can identify or spell safely."""
        allowed = set()
        for mapping_name in ("gloss_mappings",):
            allowed.update(str(value).upper() for value in self.grammar_rules.get(mapping_name, {}).values())
        for mapping_name in ("pronouns", "question_markers"):
            allowed.update(str(value).upper() for value in self.grammar_rules.get("special_cases", {}).get(mapping_name, {}).values())
        avatar_map = Path(self.grammar_rules_path).parent / "avatar_gloss_map.json"
        try:
            aliases = json.loads(avatar_map.read_text(encoding="utf-8")).get("aliases", {})
            allowed.update(str(alias).upper() for alias in aliases)
            allowed.update(str(target).upper() for target in aliases.values())
        except (OSError, json.JSONDecodeError):
            pass
        return allowed

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

    def _load_gemini(self) -> None:
        """Initialise Gemini only when explicitly enabled and keyed by the environment."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("Gemini is enabled but GEMINI_API_KEY is not configured; using rules")
            return
        try:
            from google import genai

            self.gemini_client = genai.Client(api_key=api_key)
        except Exception as error:  # Optional dependency/network configuration.
            logger.warning("Gemini is unavailable; using rules: %s", error)
            self.gemini_client = None

    def _filter_glosses(self, glosses: List[str]) -> List[str]:
        fillers = self.grammar_rules.get("filter_rules", {}).get("filler_words", [])
        filler_set = {word.upper() for word in fillers}
        return [gloss for gloss in glosses if gloss.upper() not in filler_set]

    def _apply_grammar_rules(self, glosses: List[str]) -> List[str]:
        mappings = self.grammar_rules.get("gloss_mappings", {})
        special = self.grammar_rules.get("special_cases", {})
        pronouns = special.get("pronouns", {})
        question_markers = special.get("question_markers", {})
        copulas = special.get("copulas", {})
        result = []
        for index, gloss in enumerate(glosses):
            key = gloss.upper()
            previous_key = glosses[index - 1].upper() if index else ""
            # AM is a supported copula generally; IS is only mapped in the
            # explicit "MY NAME IS <name>" introduction pattern below.
            mapped_copula = copulas.get(key, gloss) if key == "AM" or (
                key == "IS" and previous_key == "NAME"
            ) else gloss
            result.append(
                mappings.get(key, pronouns.get(key, question_markers.get(key, mapped_copula)))
            )

        # The manual-input path also accepts ordinary English-like text. For
        # the common "I am <name/description>" form, place the mapped copula
        # after its complement: "I AM NAMAN" -> "MEIN NAMAN HOON".
        for index in range(len(glosses) - 1):
            if glosses[index].upper() == "I" and glosses[index + 1].upper() == "AM":
                copula = result.pop(index + 1)
                result.append(copula)
                break
        for index in range(len(glosses) - 2):
            if [token.upper() for token in glosses[index:index + 3]] == ["MY", "NAME", "IS"]:
                copula = result.pop(index + 2)
                result.append(copula)
                break
        return result

    def _sentence_override(self, glosses: List[str]) -> List[str] | None:
        """Return a curated exact-sentence translation when one is available.

        Sentence overrides are deliberately checked before generic rules and
        optional providers. They preserve corrections supplied for a demo case
        instead of letting a token-level or remote draft rewrite them.
        """
        key = " ".join(token.casefold() for token in glosses)
        candidate = self.curated_sentence_overrides.get(key)
        if not isinstance(candidate, list) or not all(isinstance(token, str) for token in candidate):
            return None
        result = [token.strip() for token in candidate if token.strip()]
        return result or None

    def _refine_with_gemini(self, filtered: List[str], rule_based: List[str]) -> List[str] | None:
        """Produce a constrained ISL draft while retaining unknown terms verbatim."""
        if self.gemini_client is None or not rule_based:
            return None
        allowed = sorted(self.gemini_allowed_glosses | {token.upper() for token in rule_based})
        schema = {
            "type": "object",
            "properties": {
                "isl_glosses": {"type": "array", "items": {"type": "string"}},
                "unsupported_source_tokens": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["isl_glosses"],
            "additionalProperties": False,
        }
        prompt = (
            "Create an AI-assisted DRAFT ISL gloss sequence from ASL glosses. "
            "Use only tokens in the allowed list. Preserve every source meaning: if no allowed "
            "ISL token expresses a source token, include its rule-based token unchanged and list it "
            "under unsupported_source_tokens for fingerspelling/fallback. Do not claim linguistic approval. "
            "Return JSON only.\n"
            f"Allowed ISL tokens: {allowed}\nASL source: {filtered}\nRule draft: {rule_based}"
        )
        try:
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                },
            )
            payload = json.loads(response.text)
            candidate = payload.get("isl_glosses")
            if not isinstance(candidate, list) or not all(isinstance(token, str) for token in candidate):
                return None
            normalized = [token.strip() for token in candidate if token.strip()]
            unsupported = payload.get("unsupported_source_tokens", [])
            if not isinstance(unsupported, list) or not all(isinstance(token, str) for token in unsupported):
                return None
            # Every output token must be known locally. Unknown source concepts are
            # retained from the rule draft, so avatar fallback remains visible.
            if not normalized or any(token.upper() not in allowed for token in normalized):
                logger.warning("Gemini returned unsupported gloss output; using rules")
                return None
            for token in unsupported:
                normalized.append(token.strip().upper())
            return normalized
        except Exception as error:
            logger.warning("Gemini refinement failed; using rules: %s", error)
            return None

    @staticmethod
    def _parse_llm_output(output: str) -> List[str]:
        line = output.strip().splitlines()[-1] if output.strip() else ""
        return [token.strip(".,!?;:\"'") for token in line.split() if token.strip(".,!?;:\"'")]

    def translate(self, asl_glosses: List[str]) -> List[str]:
        return self.translate_with_mode(asl_glosses)[0]

    def translate_with_mode(self, asl_glosses: List[str]) -> tuple[List[str], str]:
        if not asl_glosses:
            return [], "draft_rule_based"
        filtered = self._filter_glosses(asl_glosses)
        override = self._sentence_override(filtered)
        if override is not None:
            return override, "curated_sentence_override"
        rule_based = self._apply_grammar_rules(filtered)
        if self.gemini_client is not None:
            refined = self._refine_with_gemini(filtered, rule_based)
            if refined is not None:
                return refined, "gemini_constrained_draft"
        if self.model is None or self.tokenizer is None:
            return rule_based, "draft_rule_based"
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
            return parsed or rule_based, "llama_draft"
        except Exception as error:
            logger.warning("LLM translation failed; using rules: %s", error)
            return rule_based, "draft_rule_based"

    def translate_batch(self, batch_glosses: List[List[str]]) -> List[List[str]]:
        return [self.translate(glosses) for glosses in batch_glosses]

    def translate_gloss_string(self, asl_gloss_string: str) -> str:
        return " ".join(self.translate(self._tokenize_gloss_string(asl_gloss_string)))

    def translate_gloss_string_with_mode(self, asl_gloss_string: str) -> tuple[str, str]:
        glosses, mode = self.translate_with_mode(self._tokenize_gloss_string(asl_gloss_string))
        return " ".join(glosses), mode

    @staticmethod
    def _tokenize_gloss_string(gloss_string: str) -> List[str]:
        """Split manual text while ignoring punctuation surrounding a gloss.

        Glosses such as ``THANK_YOU`` retain their internal underscore; this
        only removes sentence punctuation that otherwise prevents a mapping
        such as ``HELLO,`` from matching ``HELLO``.
        """
        return [
            token.strip(string.punctuation)
            for token in gloss_string.strip().split()
            if token.strip(string.punctuation)
        ]
