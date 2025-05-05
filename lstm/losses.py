import numpy as np
from activations import Activation


class CrossEntropyLoss:
    @staticmethod
    def compute(outputs, targets):
        loss = 0
        for t in range(len(outputs)):
            probs = Activation.softmax(outputs[t])
            loss += -np.log(probs[targets[t], 0] + 1e-8)
        return loss / len(outputs)

    @staticmethod
    def backward(outputs, targets):
        grads = []
        for t in range(len(outputs)):
            dout = Activation.softmax(outputs[t])
            dout[targets[t]] -= 1
            grads.append(dout)
        return grads

    @staticmethod
    def compute_with_gradients(outputs, targets):
        """
        同时计算交叉熵损失和输出层梯度
        参数:
            outputs: 模型输出列表，每个元素为[vocab_size, 1]的numpy数组
            targets: 目标标签列表，每个元素为整数索引
        返回:
            loss: 平均交叉熵损失（标量）
            grads: 梯度列表，每个元素对应outputs的梯度[vocab_size, 1]
        """
        loss = 0.0
        grads = []

        for t in range(len(outputs)):
            output = outputs[t]  # 当前时刻输出 [vocab_size, 1]
            target_idx = targets[t]  # 当前目标标签

            # 数值稳定的softmax计算
            max_val = np.max(output)
            exp_output = np.exp(output - max_val)
            probs = exp_output / np.sum(exp_output)

            # 计算当前时刻损失（添加微小值防止log(0)）
            eps = 1e-8
            loss += -np.log(probs[target_idx, 0] + eps)

            # 计算梯度 (dL/doutput = probs - one_hot)
            dout = probs.copy()
            dout[target_idx, 0] -= 1.0
            grads.append(dout)

        # 平均损失
        loss /= len(outputs)
        return loss, grads
