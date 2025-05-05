import numpy as np
from activations import Activation


class LSTMLayer:
    def __init__(self, input_size, hidden_size):
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))
        self.hidden_size = hidden_size
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        # 合并输入和隐藏状态
        z = np.row_stack((h_prev, x))
        a = np.dot(self.W, z) + self.b

        # 分割门控信号
        f = Activation.sigmoid(a[:self.hidden_size])
        i = Activation.sigmoid(a[self.hidden_size:2 * self.hidden_size])
        o = Activation.sigmoid(a[2 * self.hidden_size:3 * self.hidden_size])
        c_candidate = Activation.tanh(a[3 * self.hidden_size:])

        # 更新细胞状态
        c = f * c_prev + i * c_candidate
        h = o * Activation.tanh(c)

        # 保存中间结果用于反向传播
        self.cache = (z, f, i, o, c_candidate, c_prev.copy())
        return h, c

    def backward(self, dh, dc, learning_rate):
        z, f, i, o, c_candidate, c_prev = self.cache

        # 计算各梯度分量
        do = dh * Activation.tanh(c_prev) * o * (1 - o)
        dc = dh * o * (1 - Activation.tanh(c_prev) ** 2) + dc
        dc_prev = dc * f
        df = dc * c_prev * f * (1 - f)
        di = dc * c_candidate * i * (1 - i)
        dc_candidate = dc * i * (1 - c_candidate ** 2)

        # 合并梯度
        da = np.concatenate((df, di, do, dc_candidate))
        dW = np.dot(da, z.T)
        db = np.sum(da, axis=1, keepdims=True)
        dz = np.dot(self.W.T, da)
        dh_prev = dz[:self.hidden_size]
        dx = dz[self.hidden_size:]

        # 参数更新
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dx, dh_prev, dc_prev