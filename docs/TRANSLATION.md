# ASL-to-ISL Translation Module

## Overview

The translation module implements a sophisticated ASL → ISL (American Sign Language → Indian Sign Language) cross-lingual gloss translation pipeline. It combines rule-based grammar transformations with a quantized LLM (Llama-2-7b-chat) for contextual refinement.

## Architecture

```
ASL Gloss Sequence
       ↓
  [Filter Filler Words]  (discourse markers, um, uh, like, etc.)
       ↓
 [Apply Grammar Rules]   (direct gloss mappings, pronouns, question markers)
       ↓
 [LLM Refinement]        (contextual translation, tense marking, negation)
       ↓
  ISL Gloss Sequence
```

## Components

### 1. Grammar Rules (`configs/grammar_rules.json`)

Defines ASL-to-ISL transformation patterns:

- **Gloss Mappings**: Direct mappings for common signs (HELLO → NAMASKAR)
- **Tense Markers**: PAST/FUTURE prefixes for ISL temporal marking
- **Classifiers**: Motion direction transformations
- **Negation**: Explicit NAH_ prefix for negated glosses
- **Pronouns & Questions**: Special-case translations
- **Filler Words**: Words to remove before translation

### 2. Translator Class (`src/translation/translator.py`)

The main `ASLtoISLTranslator` class:

```python
class ASLtoISLTranslator:
    """Cross-lingual ASL→ISL translator with grammar rules and LLM"""
    
    def __init__(
        self,
        gloss_vocab_path: Optional[str] = None,
        grammar_rules_path: Optional[str] = None,
        config_path: Optional[str] = None,
        model_id: str = "meta-llama/Llama-2-7b-chat-hf",
        quantize: bool = True
    )
    
    def translate(self, asl_glosses: List[str]) -> List[str]:
        """Translate a sequence of ASL glosses to ISL"""
    
    def translate_batch(self, batch_glosses: List[List[str]]) -> List[List[str]]:
        """Translate multiple gloss sequences"""
    
    def translate_gloss_string(self, asl_gloss_string: str) -> str:
        """Translate space-separated gloss string"""
```

**Key Features:**
- Loads ASL gloss vocabulary from recognition model
- Applies grammar transformation rules in pipeline
- Falls back to rule-based translation if LLM fails
- Configurable temperature and top_p for sampling diversity
- Batch translation support for efficiency

### 3. Configuration (`configs/translation.yaml`)

```yaml
model:
  name: quantized_llm
  model_id: meta-llama/Llama-2-7b-chat-hf
  quantize: true          # Use 4-bit quantization
  max_length: 128         # Max output tokens
  temperature: 0.7        # Sampling temperature (0.0-1.0)
  top_p: 0.9              # Nucleus sampling threshold

data:
  batch_size: 16          # Batch size for batch_translate
  gloss_vocab_path: models/recognition/gloss_vocab.json

evaluation:
  metric: bleu            # BLEU-4 for translation quality
```

### 4. Evaluation Script (`scripts/eval_translation.py`)

Measures translation quality using multiple metrics:

- **BLEU-4**: N-gram overlap (Papineni et al., 2002)
- **chrF**: Character-level F-score (Popović, 2015)
- **TER**: Translation Edit Rate / Levenshtein distance (Snover et al., 2006)

Usage:
```bash
python scripts/eval_translation.py \
    --test-set data/translation_pairs_test.json \
    --output models/translation/eval_results.json
```

## API Integration

The translation module is integrated into the FastAPI server via the `/translate` endpoint:

```
POST /translate
Content-Type: application/json

{
  "asl_gloss": "HELLO MY NAME IS JOHN"
}

Response:
{
  "asl_gloss": "HELLO MY NAME IS JOHN",
  "isl_gloss": "NAMASKAR MERA NAM JOHN HAI",
  "confidence": 0.85
}
```

## Usage Examples

### Basic Translation

```python
from src.translation.translator import ASLtoISLTranslator

translator = ASLtoISLTranslator()

# Single sequence
asl = ["HELLO", "MY", "NAME"]
isl = translator.translate(asl)
# Returns: ["NAMASKAR", "MERA", "NAM"]

# Space-separated string
asl_str = "HELLO MY NAME"
isl_str = translator.translate_gloss_string(asl_str)
# Returns: "NAMASKAR MERA NAM"
```

### Batch Translation

```python
batch_asl = [
    ["HELLO", "MY", "NAME"],
    ["I", "LIKE", "WATER"],
    ["THANK", "YOU"]
]

results = translator.translate_batch(batch_asl)
# Returns: [
#     ["NAMASKAR", "MERA", "NAM"],
#     ["MAIN", "PASAND", "PANI"],
#     ["SHUKRIYA"]
# ]
```

### From FastAPI

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"asl_gloss": "HELLO MY NAME"}'
```

## Translation Pipeline Details

### Step 1: Filter Filler Words

Removes discourse markers and filler words that don't carry semantic meaning:
- UM, UH, LIKE, SO, WELL, YOU_KNOW
- I_MEAN, BASICALLY, LITERALLY, RIGHT

### Step 2: Apply Grammar Rules

**Direct Gloss Mapping**: Pre-defined vocabulary pairs
```
HELLO → NAMASKAR
WATER → PANI
FRIEND → DOST
```

**Pronouns**: Language-specific pronoun translation
```
I → MAIN
YOU → TUM
HE/SHE/IT → WO (gender-neutral)
```

**Question Markers**: Interrogative word mapping
```
WHO → KAUN
WHAT → KYA
WHERE → KAHAN
```

### Step 3: LLM Refinement (Optional)

If Llama-2 model is available, the translator uses an LLM-based prompt:

```
You are an expert translator between American Sign Language (ASL) and Indian Sign Language (ISL).

Given an ASL gloss sequence, translate it to ISL glosses following these rules:
1. Maintain grammatical agreement (subject, verb, object order may differ)
2. Apply tense marking when necessary (PAST_, FUTURE_ prefixes for ISL)
3. Use classifiers appropriately (motion direction, object movement)
4. Mark negation explicitly (NAH_ prefix)
5. Preserve meaning and intent

ASL Gloss Sequence: HELLO MY NAME

ISL Translation (glosses only, space-separated):
```

The LLM generates contextually appropriate ISL glosses.

## Fallback Behavior

If the LLM is unavailable or fails:
1. The translator uses rule-based output as fallback
2. Grammar rules are applied without LLM refinement
3. Graceful degradation ensures translation always completes

## Configuration Options

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `temperature` | 0.7 | 0.0-1.0 | Controls randomness (higher = more diverse) |
| `top_p` | 0.9 | 0.0-1.0 | Nucleus sampling threshold |
| `max_length` | 128 | 1-512 | Maximum output token length |
| `batch_size` | 16 | 1-∞ | Batch processing size |
| `quantize` | true | - | Load model in 4-bit quantization |

## Performance Characteristics

- **Speed**: ~200-500ms per sequence on GPU (with LLM)
- **Latency**: <100ms with rule-based fallback
- **Memory**: ~7GB (4-bit quantized Llama-2)
- **Throughput**: 16 sequences/sec (batch mode)

## Limitations & Future Improvements

### Current Limitations
1. Grammar rules are static (not learned)
2. Limited contextual understanding (LLM uses prompts only)
3. No ASL-ISL bilingual training data (evaluation uses synthetic pairs)
4. LLM model requires GPU memory

### Planned Improvements
1. Collect paired ASL-ISL corpus for fine-tuning
2. Implement trainable grammar rules module
3. Add cross-lingual BERT embeddings for better alignment
4. Support CPU-only inference with distillation
5. Real-time streaming translation

## Related Files

- `configs/grammar_rules.json` - Grammar transformation rules
- `configs/translation.yaml` - Configuration
- `configs/isl_vocabulary.json` - ISL gloss vocabulary
- `src/api/server.py` - API endpoint integration
- `src/web/app.js` - Frontend integration
- `scripts/eval_translation.py` - Evaluation framework

## References

- Papineni et al. (2002): BLEU: a Method for Automatic Evaluation of Machine Translation
- Popović (2015): chrF: character n-gram F-score for automatic MT evaluation
- Snover et al. (2006): A Study of Translation Edit Rate for Comparing Machine Translation and Human-Written Translations

## License

Part of the ASL-ISL Translation Engine (AITE) project.
