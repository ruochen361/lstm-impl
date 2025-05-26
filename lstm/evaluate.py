import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import pos_tag

def evaluate(model, test_data, src_vocab, tgt_vocab, pos_evaluator=None):
    """评估模型"""
    idx_to_word = {v: k for k, v in tgt_vocab.items()}

    references = []
    hypotheses = []

    for sample in test_data:

        # 生成翻译结果（使用自回归生成）
        pred_ids = model.generate(sample['src'], max_len=50)

        # 转换索引到词汇并过滤特殊符号
        pred_words = [idx_to_word[i] for i in pred_ids
                      if idx_to_word[i] not in {'<sos>', '<pad>', '<eos>'}]
        ref_words = [idx_to_word[i] for i in sample['tgt']
                     if idx_to_word[i] not in {'<sos>', '<pad>', '<eos>'}]

        references.append([ref_words])  # 保持嵌套结构
        hypotheses.append(pred_words)

        # 更新词性混淆矩阵
        if pos_evaluator is not None:
            pos_evaluator.update([ref_words], [pred_words])

        # 使用标准BLEU计算
        return corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method4)




class TranslationEvaluator:
    def __init__(self, top_k_words=50):
        self.word_counts = defaultdict(int)
        self.confusion_dict = defaultdict(lambda: defaultdict(int))
        self.top_k = top_k_words

    def update(self, references, hypotheses):
        """更新统计信息"""
        for ref_seq, hyp_seq in zip(references, hypotheses):
            min_len = min(len(ref_seq), len(hyp_seq))
            for i in range(min_len):
                ref_word = ref_seq[i]
                hyp_word = hyp_seq[i]
                self.word_counts[ref_word] += 1
                self.word_counts[hyp_word] += 1
                self.confusion_dict[ref_word][hyp_word] += 1

    def get_confusion_matrix(self):
        """生成Top-K高频词混淆矩阵"""
        # 选取高频词
        top_words = sorted(self.word_counts.items(),
                           key=lambda x: -x[1])[:self.top_k]
        top_words = [w[0] for w in top_words]

        # 初始化矩阵
        size = len(top_words)
        matrix = np.zeros((size, size), dtype=int)
        word2idx = {w: i for i, w in enumerate(top_words)}

        # 填充矩阵
        for ref_word in top_words:
            if ref_word not in self.confusion_dict:
                continue
            for hyp_word, count in self.confusion_dict[ref_word].items():
                if hyp_word in word2idx:
                    matrix[word2idx[ref_word], word2idx[hyp_word]] += count

        return matrix, top_words

    def plot_matrix(self, matrix, labels):
        """可视化混淆矩阵"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=False, fmt='d',
                    xticklabels=labels, yticklabels=labels,
                    cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Word-Level Confusion Matrix (Top {} Words)'.format(self.top_k))
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()




class POSConfusionMatrix:
    def __init__(self):
        self.tag_set = {'NN', 'VB', 'JJ', 'IN', 'DT'}  # 定义关注的词性类别
        self.matrix = np.zeros((len(self.tag_set), len(self.tag_set)), dtype=int)
        self.tag2idx = {tag: i for i, tag in enumerate(sorted(self.tag_set))}

    def _get_pos(self, word):
        """获取单词的主词性"""
        tagged = pos_tag([word])
        pos = tagged[0][1][:2]  # 取简写形式
        return pos if pos in self.tag_set else 'OTHER'

    def update(self, references, hypotheses):
        """更新矩阵"""
        for ref_seq, hyp_seq in zip(references, hypotheses):
            min_len = min(len(ref_seq), len(hyp_seq))
            for i in range(min_len):
                ref_pos = self._get_pos(ref_seq[i])
                hyp_pos = self._get_pos(hyp_seq[i])
                if ref_pos in self.tag2idx and hyp_pos in self.tag2idx:
                    row = self.tag2idx[ref_pos]
                    col = self.tag2idx[hyp_pos]
                    self.matrix[row, col] += 1

    def reset(self):
        self.matrix.fill(0)
    def plot(self):
        """可视化"""
        labels = sorted(self.tag_set)
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.matrix, annot=True, fmt='d',
                    xticklabels=labels, yticklabels=labels,
                    cmap='Greens')
        plt.title("POS Tag Confusion Matrix")
        plt.xlabel("Predicted POS")
        plt.ylabel("True POS")
        plt.show()
