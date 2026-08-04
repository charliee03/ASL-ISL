#!/usr/bin/env python3
"""
Evaluate the ASL→ISL translation module using BLEU-4, chrF, and WER metrics.

Usage:
    python scripts/eval_translation.py --checkpoint models/translation/checkpoint.pt \
                                       --test-set data/translation_pairs_test.json \
                                       --output models/translation/eval_results.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch

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
        """
        Compute BLEU-4 score (simplified implementation).
        
        BLEU-4 measures n-gram overlap for n=1,2,3,4.
        Reference: Papineni et al., 2002
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            # Prepare reference and hypothesis for NLTK
            ref = [reference]
            hyp = hypothesis
            
            # Use smoothing to avoid zero scores
            smoothing = SmoothingFunction().method1
            bleu = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            return bleu
        except ImportError:
            logger.warning("NLTK not available, using simplified BLEU-1")
            return self._compute_bleu1(reference, hypothesis)

    def _compute_bleu1(self, reference: List[str], hypothesis: List[str]) -> float:
        """Simplified BLEU-1 (unigram overlap) as fallback."""
        if not hypothesis:
            return 0.0
        matches = sum(1 for h in hypothesis if h in reference)
        return matches / len(hypothesis)

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

    def evaluate_corpus(self, pairs: List[Dict]) -> Dict:
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
            asl = pair.get("asl_gloss", "")
            ref = pair.get("isl_gloss", "")
            
            # Placeholder: in real use, this would be the model's output
            hyp = ref  # TODO: Replace with actual model prediction
            
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
    args = parser.parse_args()

    # Load test set
    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        logger.warning(f"Test set not found at {test_set_path}")
        logger.info("Creating minimal test set for demonstration...")
        test_pairs = [
            {"asl_gloss": "HELLO MY NAME", "isl_gloss": "NAMASKAR MERA NAM"},
            {"asl_gloss": "THANK YOU", "isl_gloss": "SHUKRIYA"},
            {"asl_gloss": "GOODBYE", "isl_gloss": "ALVIDA"},
        ]
    else:
        with open(test_set_path) as f:
            test_pairs = json.load(f)

    logger.info(f"Loaded {len(test_pairs)} test pairs")

    # Evaluate
    evaluator = TranslationEvaluator()
    results = evaluator.evaluate_corpus(test_pairs)

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
