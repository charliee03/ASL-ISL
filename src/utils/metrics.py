from sacrebleu.metrics import BLEU
from jiwer import wer


class Metrics:
    def __init__(self):
        # Keep evaluation offline and reproducible.  Hugging Face's `evaluate`
        # loader downloads metric scripts at runtime, which breaks local demos
        # and isolated deployment environments.
        self.bleu = BLEU(effective_order=True)

    def compute_bleu(self, predictions, references):
        return {"score": self.bleu.corpus_score(predictions, references).score}

    def compute_wer(self, predictions, references):
        return wer(references, predictions)
