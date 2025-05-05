import numpy as np
from layer import LSTMLayer


class Encoder:
    def __init__(self, input_size, hidden_size):
        self.lstm = LSTMLayer(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, src_seq):
        """
        编码器前向传播
        参数:
            src_seq: 源语言序列 [seq_len, input_size, 1]
        返回:
            hidden_state: 最后时刻的隐藏状态 [hidden_size, 1]
            cell_state: 最后时刻的细胞状态 [hidden_size, 1]
        """
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        # 处理整个输入序列
        for x in src_seq:
            h, c = self.lstm.forward(x, h, c)
        return h, c

    def get_params(self):
        return self.lstm.W, self.lstm.b

    def set_params(self, W, b):
        self.lstm.W = W
        self.lstm.b = b