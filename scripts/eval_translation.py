#!/usr/bin/env python3
"""Evaluate real ASL→ISL translator predictions against a held-out test set."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.translation.translator import ASLtoISLTranslator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TranslationEvaluator:
    """Evaluate ASL→ISL translation quality using reference ISL glosses."""

    def __init__(self):
        self.results = {
            "bleu4": [],
            "chrf": [],
            "ter": [],
            "samples": []
        }

    def _tokenize_gloss(self, gloss_string: str) -> List[str]:
        """Tokenize a gloss string into words."""
        return gloss_string.upper().split()

    def _compute_bleu4(self, reference: List[str], hypothesis: List[str]) -> float:
        """Compute sentence BLEU-4 with SacreBLEU, the project dependency."""
        from sacrebleu.metrics import BLEU

        # effective_order keeps the metric defined for short gloss sequences
        # while still computing up to four-gram precision where possible.
        score = BLEU(effective_order=True).corpus_score(
            [" ".join(hypothesis)], [[" ".join(reference)]]
        )
        return score.score / 100.0

    def _compute_chrf(self, reference: str, hypothesis: str) -> float:
        """
        Compute chrF score (character n-gram F-score).
        
        Measures character-level n-gram overlap, useful for morphologically rich languages.
        Reference: Popović, 2015
        """
        try:
            from sacrebleu import CHRF
            chrf_metric = CHRF()
            score = chrf_metric.corpus_score([hypothesis], [[reference]])
            return score.score / 100.0  # Normalize to [0,1]
        except ImportError:
            logger.warning("sacrebleu not available, using fallback char overlap")
            # Simple character-level overlap as fallback
            ref_chars = set(reference.lower())
            hyp_chars = set(hypothesis.lower())
            if not hyp_chars:
                return 0.0
            overlap = len(ref_chars & hyp_chars)
            return overlap / len(hyp_chars)

    def _compute_ter(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        Compute Translation Edit Rate (TER).
        
        TER measures the number of edits (insertions, deletions, substitutions, shifts)
        needed to transform hypothesis into reference.
        Reference: Snover et al., 2006
        """
        # Simplified TER: Levenshtein distance normalized by reference length
        ref_len = len(reference)
        if ref_len == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
        
        # Compute Levenshtein distance (minimum edits)
        hyp_len = len(hypothesis)
        dp = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
        
        for i in range(ref_len + 1):
            dp[i][0] = i
        for j in range(hyp_len + 1):
            dp[0][j] = j
        
        for i in range(1, ref_len + 1):
            for j in range(1, hyp_len + 1):
                if reference[i-1] == hypothesis[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[ref_len][hyp_len] / ref_len

    def evaluate_pair(
        self,
        asl_gloss: str,
        reference_isl: str,
        hypothesis_isl: str
    ) -> Dict[str, float]:
        """
        Evaluate a single translation pair.

        Args:
            asl_gloss: Input ASL gloss string
            reference_isl: Reference ISL gloss string
            hypothesis_isl: Predicted ISL gloss string

        Returns:
            Dictionary with BLEU-4, chrF, and TER scores
        """
        ref_tokens = self._tokenize_gloss(reference_isl)
        hyp_tokens = self._tokenize_gloss(hypothesis_isl)

        bleu4 = self._compute_bleu4(ref_tokens, hyp_tokens)
        chrf = self._compute_chrf(reference_isl.upper(), hypothesis_isl.upper())
        ter = self._compute_ter(ref_tokens, hyp_tokens)

        return {
            "bleu4": bleu4,
            "chrf": chrf,
            "ter": ter,
            "asl_gloss": asl_gloss,
            "reference_isl": reference_isl,
            "hypothesis_isl": hypothesis_isl,
        }

    def evaluate_corpus(self, pairs: List[Dict], translator: ASLtoISLTranslator) -> Dict:
        """
        Evaluate translation quality on a corpus of ASL-ISL pairs.

        Args:
            pairs: List of dicts with keys: asl_gloss, isl_gloss

        Returns:
            Dictionary with corpus-level metrics
        """
        bleu_scores = []
        chrf_scores = []
        ter_scores = []
        all_samples = []

        for pair in pairs:
            # Support the current API-style keys and the paired-data keys used
            # by validate_pairs.py, without silently accepting incomplete rows.
            asl = str(pair.get("asl_gloss", pair.get("asl", ""))).strip()
            ref = str(pair.get("isl_gloss", pair.get("isl", ""))).strip()
            if not asl or not ref:
                raise ValueError("Each test pair must include non-empty ASL and ISL glosses")

            hyp = translator.translate_gloss_string(asl)
            
            metrics = self.evaluate_pair(asl, ref, hyp)
            
            bleu_scores.append(metrics["bleu4"])
            chrf_scores.append(metrics["chrf"])
            ter_scores.append(metrics["ter"])
            all_samples.append(metrics)

        corpus_metrics = {
            "bleu4": {
                "mean": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
                "max": max(bleu_scores) if bleu_scores else 0.0,
                "min": min(bleu_scores) if bleu_scores else 0.0,
            },
            "chrf": {
                "mean": sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0,
                "max": max(chrf_scores) if chrf_scores else 0.0,
                "min": min(chrf_scores) if chrf_scores else 0.0,
            },
            "ter": {
                "mean": sum(ter_scores) / len(ter_scores) if ter_scores else 0.0,
                "max": max(ter_scores) if ter_scores else 0.0,
                "min": min(ter_scores) if ter_scores else 0.0,
            },
            "samples": all_samples[:10],  # Show first 10 samples
            "total_pairs": len(pairs),
        }

        return corpus_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASL→ISL translation module"
    )
    parser.add_argument(
        "--test-set",
        default="data/translation_pairs_test.json",
        help="Path to test set with ASL-ISL pairs"
    )
    parser.add_argument(
        "--output",
        default="models/translation/eval_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Use the configured Hugging Face model instead of deterministic rules"
    )
    args = parser.parse_args()

    # Load test set
    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        raise FileNotFoundError(
            f"Held-out test set not found: {test_set_path}. "
            "Create and validate it before reporting BLEU or chrF metrics."
        )
    with open(test_set_path) as f:
        test_pairs = json.load(f)
    if not isinstance(test_pairs, list) or not test_pairs:
        raise ValueError("The test set must be a non-empty JSON array")

    logger.info(f"Loaded {len(test_pairs)} test pairs")

    translator = ASLtoISLTranslator(enable_llm=args.enable_llm)

    # Evaluate actual translator output; never substitute the reference as a prediction.
    evaluator = TranslationEvaluator()
    results = evaluator.evaluate_corpus(test_pairs, translator)
    results["evaluation_mode"] = "llm" if args.enable_llm else "rule_based"
    results["test_set"] = str(test_set_path)

    logger.info(f"BLEU-4 (mean): {results['bleu4']['mean']:.4f}")
    logger.info(f"chrF (mean): {results['chrf']['mean']:.4f}")
    logger.info(f"TER (mean): {results['ter']['mean']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
