import evaluate


class Metrics:
    def __init__(self):
        self.bleu = evaluate.load("sacrebleu")
        self.wer = evaluate.load("wer")

    def compute_bleu(self, predictions, references):
        return self.bleu.compute(predictions=predictions, references=references)

    def compute_wer(self, predictions, references):
        return self.wer.compute(predictions=predictions, references=references)
