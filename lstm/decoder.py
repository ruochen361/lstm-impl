import numpy as np
from layer import LSTMLayer


class Decoder:
    def __init__(self, output_size, hidden_size):
        self.lstm = LSTMLayer(output_size, hidden_size)
        self.W_out = np.random.randn(output_size, hidden_size) * 0.01
        self.b_out = np.zeros((output_size, 1))
        self.hidden_size = hidden_size

    def forward_step(self, x, h_prev, c_prev):
        """
        解码器单步前向传播
        参数:
            x: 当前时刻输入 [output_size, 1]
            h_prev: 前一时隐藏状态 [hidden_size, 1]
            c_prev: 前一时细胞状态 [hidden_size, 1]
        返回:
            output: 当前时刻输出 [output_size, 1]
            h: 新隐藏状态
            c: 新细胞状态
        """
        h, c, cache = self.lstm.forward_step(x, h_prev, c_prev)
        output = np.dot(self.W_out, h) + self.b_out
        return output, h, c, cache

    def get_params(self):
        return self.lstm.W, self.lstm.b, self.W_out, self.b_out

    def set_params(self, W, b, W_out, b_out):
        self.lstm.W = W
        self.lstm.b = b
        self.W_out = W_out
        self.b_out = b_out