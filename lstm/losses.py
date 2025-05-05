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