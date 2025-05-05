import numpy as np
from nltk.translate.bleu_score import sentence_bleu


class NMTMetrics:
    @staticmethod
    def bleu_score(references, hypotheses):
        return sentence_bleu(references, hypotheses)

    @staticmethod
    def accuracy(preds, targets):
        return np.mean(np.array(preds) == np.array(targets))