import numpy as np
from encoder import Encoder
from decoder import Decoder


class Seq2Seq:
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size):
        self.encoder = Encoder(src_vocab_size, hidden_size)
        self.decoder = Decoder(tgt_vocab_size, hidden_size)

    def forward(self, src_seq, tgt_seq, teacher_forcing_ratio=0.5):
        """
        完整前向传播流程
        参数:
            src_seq: 源序列列表 [seq_len, vocab_size, 1]
            tgt_seq: 目标序列列表 [seq_len]
            teacher_forcing_ratio: 使用真实标签的概率
        返回:
            outputs: 各时刻输出 [seq_len, vocab_size]
        """
        # 编码阶段
        h_enc, c_enc, _ = self.encoder.forward(src_seq)

        # 解码阶段初始化
        seq_len = len(tgt_seq)
        outputs = []
        h_dec, c_dec = h_enc, c_enc
        prev_token = np.zeros((self.decoder.lstm.input_size, 1))  # 初始输入为0

        for t in range(seq_len):
            # 决定是否使用教师强制
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            if use_teacher_forcing and t > 0:
                prev_token = np.zeros_like(prev_token)
                # 使用真实上一时刻标签
                prev_token[tgt_seq[t - 1]] = 1.0

            # 解码步骤
            output, h_dec, c_dec, dec_cache = self.decoder.forward_step(prev_token, h_dec, c_dec)
            outputs.append(output)

            # 更新下一时刻输入
            prev_token = np.zeros_like(prev_token)
            prev_token[np.argmax(output)] = 1.0  # 使用预测结果

        return outputs

    def get_params(self):
        enc_W, enc_b = self.encoder.get_params()
        dec_W, dec_b, dec_W_out, dec_b_out = self.decoder.get_params()
        return {
            'encoder.W': enc_W,
            'encoder.b': enc_b,
            'decoder.W': dec_W,
            'decoder.b': dec_b,
            'decoder.W_out': dec_W_out,
            'decoder.b_out': dec_b_out
        }

    def set_params(self, params):
        self.encoder.set_params(params['encoder.W'], params['encoder.b'])
        self.decoder.set_params(
            params['decoder.W'],
            params['decoder.b'],
            params['decoder.W_out'],
            params['decoder.b_out']
        )