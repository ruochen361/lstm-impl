import numpy as np
from layer import LSTMLayer


class Encoder:
    def __init__(self, input_size, hidden_size):
        self.lstm = LSTMLayer(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.forward_cache = []  # 新增缓存存储

    def forward(self, src_seq):
        """
        编码器前向传播（带缓存记录）
        参数:
            src_seq: 源语言序列 [seq_len, input_size, 1]
        返回:
            hidden_state: 最后时刻的隐藏状态 [hidden_size, 1]
            cell_state: 最后时刻的细胞状态 [hidden_size, 1]
            cache_list: 各时间步的缓存列表
        """
        self.forward_cache = []  # 重置缓存
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        for x in src_seq:
            h, c, cache = self.lstm.forward_step(x, h, c)  # 使用forward_step获取缓存
            self.forward_cache.append((x, h.copy(), c.copy(), cache))

        return h, c, self.forward_cache  # 返回缓存列表

    def get_params(self):
        return self.lstm.W, self.lstm.b

    def set_params(self, W, b):
        self.lstm.W = W
        self.lstm.b = b