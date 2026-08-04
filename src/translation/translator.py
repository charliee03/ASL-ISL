import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ASLtoISLTranslator:
    """
    Cross-lingual ASL → ISL translator with grammar rules and LLM backbone.
    
    Pipeline:
    1. Load ASL gloss vocabulary from recognition model
    2. Apply grammar transformation rules (subject-verb order, tense marking, etc.)
    3. Filter filler words and discourse markers
    4. Use quantized LLM (Llama-2-7b-chat) for contextual translation
    5. Output ISL glosses
    """

    def __init__(
        self,
        gloss_vocab_path: Optional[str] = None,
        grammar_rules_path: Optional[str] = None,
        config_path: Optional[str] = None,
        model_id: str = "meta-llama/Llama-2-7b-chat-hf",
        quantize: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the ASL→ISL translator.

        Args:
            gloss_vocab_path: Path to gloss vocabulary JSON from recognition model
            grammar_rules_path: Path to grammar transformation rules JSON
            config_path: Path to translation config YAML
            model_id: Hugging Face model ID (default: Llama-2-7b-chat)
            quantize: Whether to load model in 4-bit quantization
            device: Device to load model on (default: auto-detect)
        """
        self.gloss_vocab_path = gloss_vocab_path or "models/recognition/gloss_vocab.json"
        self.grammar_rules_path = grammar_rules_path or "configs/grammar_rules.json"
        self.config_path = config_path or "configs/translation.yaml"
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load configurations
        self.gloss_vocab = self._load_gloss_vocab()
        self.grammar_rules = self._load_grammar_rules()
        self.config = self._load_config()

        # Initialize the optional LLM.  Keep the rule-based translator usable
        # when model weights are unavailable (for example, in an offline demo).
        self.tokenizer = None
        self.model = None
        try:
            logger.info(f"Loading tokenizer from {model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Loading model from {model_id} (quantize={quantize})...")
            if quantize:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    load_in_4bit=True,
                    device_map=self.device,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=self.device,
                )
            self.model.eval()
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Translator will fall back to rule-based translation only")
            self.tokenizer = None
            self.model = None

        # Hyperparameters from config
        self.max_length = self.config.get("model", {}).get("max_length", 128)
        self.temperature = self.config.get("model", {}).get("temperature", 0.7)
        self.top_p = self.config.get("model", {}).get("top_p", 0.9)
        self.batch_size = self.config.get("data", {}).get("batch_size", 16)

    def _load_gloss_vocab(self) -> Dict[str, int]:
        """Load ASL gloss vocabulary from recognition model."""
        try:
            path = Path(self.gloss_vocab_path)
            if not path.exists():
                logger.warning(f"Gloss vocab not found at {path}, using empty vocab")
                return {}
            
            with open(path) as f:
                data = json.load(f)
                # Recognition exports a metadata wrapper with both mappings.
                if isinstance(data, dict) and isinstance(data.get("gloss_to_id"), dict):
                    return data["gloss_to_id"]
                if isinstance(data, dict) and isinstance(data.get("id_to_gloss"), dict):
                    return {gloss: int(index) for index, gloss in data["id_to_gloss"].items()}
                # Also support the two compact formats: {gloss: id} and {id: gloss}.
                if not data:
                    return {}
                first_value = next(iter(data.values()))
                if isinstance(first_value, int):
                    return data
                return {gloss: index for index, gloss in data.items()}
        except Exception as e:
            logger.error(f"Error loading gloss vocab: {e}")
            return {}

    def _load_grammar_rules(self) -> Dict:
        """Load grammar transformation rules."""
        try:
            path = Path(self.grammar_rules_path)
            if not path.exists():
                logger.warning(f"Grammar rules not found at {path}, using empty rules")
                return {"gloss_mappings": {}, "filter_rules": {}}
            
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading grammar rules: {e}")
            return {"gloss_mappings": {}, "filter_rules": {}}

    def _load_config(self) -> Dict:
        """Load translation configuration."""
        try:
            import yaml
            path = Path(self.config_path)
            if not path.exists():
                logger.warning(f"Config not found at {path}, using defaults")
                return {"model": {}, "data": {}, "training": {}}
            
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {"model": {}, "data": {}, "training": {}}

    def _filter_glosses(self, glosses: List[str]) -> List[str]:
        """Remove filler words and discourse markers."""
        filter_list = self.grammar_rules.get("filter_rules", {}).get("filler_words", [])
        filter_set = set(word.upper() for word in filter_list)
        return [g for g in glosses if g.upper() not in filter_set]

    def _apply_grammar_rules(self, glosses: List[str]) -> List[str]:
        """Apply grammar transformation rules (rule-based layer)."""
        mappings = self.grammar_rules.get("gloss_mappings", {})
        transformed = []

        for gloss in glosses:
            # Direct gloss mapping (e.g., HELLO → NAMASKAR)
            if gloss.upper() in mappings:
                transformed.append(mappings[gloss.upper()])
            # Special case: pronouns
            elif gloss.upper() in self.grammar_rules.get("special_cases", {}).get("pronouns", {}):
                pronouns = self.grammar_rules["special_cases"]["pronouns"]
                transformed.append(pronouns[gloss.upper()])
            # Special case: question markers
            elif gloss.upper() in self.grammar_rules.get("special_cases", {}).get("question_markers", {}):
                markers = self.grammar_rules["special_cases"]["question_markers"]
                transformed.append(markers[gloss.upper()])
            else:
                # Fallback: keep original gloss
                transformed.append(gloss)

        return transformed

    def _build_translation_prompt(self, asl_glosses: List[str]) -> str:
        """
        Build a prompt for the LLM to translate ASL glosses to ISL.
        
        The prompt guides the model to:
        - Understand the ASL gloss sequence
        - Apply cross-lingual grammar transformations
        - Output ISL glosses
        """
        asl_text = " ".join(asl_glosses)
        
        prompt = f"""You are an expert translator between American Sign Language (ASL) and Indian Sign Language (ISL).

Given an ASL gloss sequence, translate it to ISL glosses following these rules:
1. Maintain grammatical agreement (subject, verb, object order may differ)
2. Apply tense marking when necessary (PAST_, FUTURE_ prefixes for ISL)
3. Use classifiers appropriately (motion direction, object movement)
4. Mark negation explicitly (NAH_ prefix)
5. Preserve meaning and intent

ASL Gloss Sequence:
{asl_text}

ISL Translation (glosses only, space-separated):"""

        return prompt

    def _parse_llm_output(self, output: str) -> List[str]:
        """Parse LLM output into ISL glosses."""
        # Extract the last line (model's response)
        lines = output.strip().split("\n")
        isl_line = lines[-1].strip()
        
        # Split into glosses and clean
        isl_glosses = isl_line.split()
        
        # Remove any trailing punctuation or special characters
        isl_glosses = [g.strip(".,!?;:\"'") for g in isl_glosses]
        
        return isl_glosses

    def translate(self, asl_glosses: List[str]) -> List[str]:
        """
        Translate a sequence of ASL glosses to ISL.

        Args:
            asl_glosses: List of ASL gloss strings (e.g., ["HELLO", "MY", "NAME"])

        Returns:
            List of ISL gloss strings
        """
        if not asl_glosses:
            return []

        # Step 1: Filter filler words
        filtered = self._filter_glosses(asl_glosses)
        if not filtered:
            return []

        # Step 2: Apply grammar rules (rule-based layer)
        rule_based = self._apply_grammar_rules(filtered)

        # Step 3: Use LLM for contextual refinement (if available)
        if self.model is not None:
            try:
                prompt = self._build_translation_prompt(filtered)
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                isl_glosses = self._parse_llm_output(decoded)
                
                # Fallback to rule-based if LLM output is empty or malformed
                return isl_glosses if isl_glosses else rule_based
            except Exception as e:
                logger.warning(f"LLM translation failed: {e}, falling back to rule-based")
                return rule_based
        else:
            # No LLM available, return rule-based translation
            return rule_based

    def translate_batch(self, batch_glosses: List[List[str]]) -> List[List[str]]:
        """
        Translate multiple sequences of ASL glosses.

        Args:
            batch_glosses: List of gloss sequences

        Returns:
            List of ISL gloss sequences
        """
        results = []
        for glosses in batch_glosses:
            isl_glosses = self.translate(glosses)
            results.append(isl_glosses)
        return results

    def translate_gloss_string(self, asl_gloss_string: str) -> str:
        """
        Translate a space-separated gloss string.

        Args:
            asl_gloss_string: Space-separated ASL glosses (e.g., "HELLO MY NAME")

        Returns:
            Space-separated ISL glosses
        """
        asl_glosses = asl_gloss_string.split()
        isl_glosses = self.translate(asl_glosses)
        return " ".join(isl_glosses)
