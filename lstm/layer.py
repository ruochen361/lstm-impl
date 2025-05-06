import numpy as np
from activations import Activation


class LSTMLayer:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))
        self.hidden_size = hidden_size
        self.cache = None

    # def forward(self, x, h_prev, c_prev):
    #     # 合并输入和隐藏状态
    #     z = np.row_stack((h_prev, x))
    #     a = np.dot(self.W, z) + self.b
    #
    #     # 分割门控信号
    #     f = Activation.sigmoid(a[:self.hidden_size])
    #     i = Activation.sigmoid(a[self.hidden_size:2 * self.hidden_size])
    #     o = Activation.sigmoid(a[2 * self.hidden_size:3 * self.hidden_size])
    #     c_candidate = Activation.tanh(a[3 * self.hidden_size:])
    #
    #     # 更新细胞状态
    #     c = f * c_prev + i * c_candidate
    #     h = o * Activation.tanh(c)
    #
    #     # 保存中间结果用于反向传播
    #     self.cache = (z, f, i, o, c_candidate, c, h_prev, c_prev)
    #     return h, c, self.cache
    #
    # def backward(self, dh, dc, learning_rate):
    #     z, f, i, o, c_candidate, c_prev = self.cache
    #
    #     # 计算各梯度分量
    #     do = dh * Activation.tanh(c_prev) * o * (1 - o)
    #     dc = dh * o * (1 - Activation.tanh(c_prev) ** 2) + dc
    #     dc_prev = dc * f
    #     df = dc * c_prev * f * (1 - f)
    #     di = dc * c_candidate * i * (1 - i)
    #     dc_candidate = dc * i * (1 - c_candidate ** 2)
    #
    #     # 合并梯度
    #     da = np.concatenate((df, di, do, dc_candidate))
    #     dW = np.dot(da, z.T)
    #     db = np.sum(da, axis=1, keepdims=True)
    #     dz = np.dot(self.W.T, da)
    #     dh_prev = dz[:self.hidden_size]
    #     dx = dz[self.hidden_size:]
    #
    #     # 参数更新
    #     self.W -= learning_rate * dW
    #     self.b -= learning_rate * db
    #
    #     return dx, dh_prev, dc_prev

    def forward_step(self, x, h_prev, c_prev):
        """
        单时间步前向传播
        参数:
            x: 输入向量 [input_size, 1]
            h_prev: 前一隐藏状态 [hidden_size, 1]
            c_prev: 前一刻细胞状态 [hidden_size, 1]
        返回:
            h: 新隐藏状态
            c: 新细胞状态
            cache: 用于反向传播的元组
        """
        # 拼接输入和隐藏状态
        z = np.vstack((h_prev, x))  # [hidden_size+input_size, 1]

        # 计算门控信号
        a = np.dot(self.W, z) + self.b  # [4*hidden_size, 1]

        # 分割门控信号
        f = Activation.sigmoid(a[:self.hidden_size])
        i = Activation.sigmoid(a[self.hidden_size:2 * self.hidden_size])
        o = Activation.sigmoid(a[2 * self.hidden_size:3 * self.hidden_size])
        c_candidate = Activation.tanh(a[3 * self.hidden_size:])

        # 更新细胞状态
        c = f * c_prev + i * c_candidate
        h = o * Activation.tanh(c)

        # 保存中间结果用于反向传播
        self.cache = (z, f, i, o, c_candidate, c_prev.copy(), h_prev.copy())
        return h, c, self.cache

    def backward_step(self, dh, dc, cache):
        """
        单时间步反向传播
        参数:
            dh: 来自下一时刻的隐藏状态梯度 [hidden_size, 1]
            dc: 来自下一时刻的细胞状态梯度 [hidden_size, 1]
            cache: 前向传播保存的缓存
        返回:
            dx: 输入梯度 [input_size, 1]
            dh_prev: 前一隐藏状态梯度
            dc_prev: 前一刻细胞状态梯度
            dW: 权重梯度
            db: 偏置梯度
        """
        z, f, i, o, c_candidate, c_prev, h_prev = cache
        h_size = self.hidden_size

        # 计算当前时刻的梯度分量
        do = dh * Activation.tanh(c_prev) * o * (1 - o)
        dc_current = dh * o * (1 - Activation.tanh(c_prev) ** 2) + dc

        # 计算各门控梯度
        df = dc_current * c_prev * f * (1 - f)
        di = dc_current * c_candidate * i * (1 - i)
        dc_candidate_grad = dc_current * i * (1 - c_candidate ** 2)

        # 合并梯度
        da = np.vstack((df, di, do, dc_candidate_grad))  # [4*hidden_size, 1]

        # 计算参数梯度
        dW = np.dot(da, z.T)  # [4h, h+i] · [h+i,1]^T → [4h, h+i]
        db = da  # [4h, 1]

        # 计算输入梯度
        dz = np.dot(self.W.T, da)  # [h+i,4h] · [4h,1] → [h+i,1]
        dh_prev = dz[:h_size]  # [h,1]
        dx = dz[h_size:]  # [i,1]

        # 计算前一时间步的细胞状态梯度
        dc_prev = f * dc_current

        return dx, dh_prev, dc_prev, dW, db