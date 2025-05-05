import numpy as np
from seq2seq import Seq2Seq
from losses import CrossEntropyLoss
from optimizer import Adam
from utils import NMTMetrics


# 配置参数
class Config:
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    hidden_size = 128
    learning_rate = 0.001
    batch_size = 32
    epochs = 10
    max_seq_len = 20


def generate_dummy_data(config):
    """生成虚拟训练数据"""
    data = []
    for _ in range(100):  # 100个样本
        seq_len = np.random.randint(5, config.max_seq_len)
        src = [np.random.rand(config.src_vocab_size, 1) for _ in range(seq_len)]
        tgt = [np.random.randint(config.tgt_vocab_size) for _ in range(seq_len)]
        data.append({'src': src, 'tgt': tgt})
    return data


def train():
    # 初始化组件
    model = Seq2Seq(Config.src_vocab_size, Config.tgt_vocab_size, Config.hidden_size)
    optimizer = Adam(lr=Config.learning_rate)
    train_data = generate_dummy_data(Config)

    # 训练循环
    for epoch in range(Config.epochs):
        total_loss = 0
        for batch in train_data:
            # 前向传播
            outputs = model.forward(batch['src'], batch['tgt'])

            # 计算损失和输出梯度
            loss, output_grads = CrossEntropyLoss.compute_with_gradients(outputs, batch['tgt'])

            # 执行BPTT
            grads = backpropagate_through_time(model, batch['src'], batch['tgt'], output_grads)

            # 获更新梯度
            params = model.get_params()
            optimizer.update(params, grads)

            total_loss += loss

        # 验证评估
        avg_loss = total_loss / len(train_data)
        bleu = evaluate(model)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | BLEU: {bleu:.4f}")


def calculate_gradients(model, output_grads):
    """计算各参数梯度（简化示例）"""
    gradients = {}

    # 解码器输出层梯度
    decoder_W_grad = np.zeros_like(model.decoder.W_out)
    decoder_b_grad = np.zeros_like(model.decoder.b_out)

    for t, grad in enumerate(output_grads):
        h_dec = model.decoder.lstm.hidden_state_history[t]
        decoder_W_grad += np.outer(grad, h_dec)
        decoder_b_grad += grad

    gradients['decoder.W_out'] = decoder_W_grad / len(output_grads)
    gradients['decoder.b_out'] = decoder_b_grad.mean(axis=1, keepdims=True)

    # LSTM梯度需要根据时间步传播（此处简化实现）
    # 实际需要实现BPTT算法
    gradients['encoder.W'] = np.random.randn(*model.encoder.lstm.W.shape) * 0.01
    gradients['encoder.b'] = np.random.randn(*model.encoder.lstm.b.shape) * 0.01
    gradients['decoder.W'] = np.random.randn(*model.decoder.lstm.W.shape) * 0.01
    gradients['decoder.b'] = np.random.randn(*model.decoder.lstm.b.shape) * 0.01

    backpropagate_through_time(model, src_seq, tgt_seq, output_grads)
    return gradients


def backpropagate_through_time(model, src_seq, tgt_seq, output_grads):
    """
    完整的BPTT实现
    参数:
        model: Seq2Seq模型实例
        src_seq: 源序列 [seq_len, input_size, 1]
        tgt_seq: 目标序列 [seq_len]
        output_grads: 输出层梯度列表 [seq_len, output_size, 1]
    返回:
        grads: 包含所有参数的梯度字典
    """
    # 初始化梯度容器
    grads = {
        'encoder.W': np.zeros_like(model.encoder.lstm.W),
        'encoder.b': np.zeros_like(model.encoder.lstm.b),
        'decoder.W': np.zeros_like(model.decoder.lstm.W),
        'decoder.b': np.zeros_like(model.decoder.lstm.b),
        'decoder.W_out': np.zeros_like(model.decoder.W_out),
        'decoder.b_out': np.zeros_like(model.decoder.b_out)
    }

    # 前向传播并保存所有中间状态
    encoder_states = []
    h_enc, c_enc = np.zeros((model.encoder.hidden_size, 1)), np.zeros((model.encoder.hidden_size, 1))
    for x in src_seq:
        h_enc, c_enc, enc_cache = model.encoder.lstm.forward_step(x, h_enc, c_enc)
        encoder_states.append((x, h_enc.copy(), c_enc.copy(), enc_cache))

    # 解码器前向传播保存状态
    decoder_states = []
    h_dec, c_dec = h_enc.copy(), c_enc.copy()
    for t in range(len(tgt_seq)):
        x = np.zeros((model.decoder.lstm.input_size, 1))  # 假设使用基础输入
        h_dec, c_dec, dec_cache = model.decoder.lstm.forward_step(x, h_dec, c_dec)
        decoder_states.append(dec_cache)

    # 反向传播解码器
    dh_next = np.zeros_like(h_dec)
    dc_next = np.zeros_like(c_dec)

    for t in reversed(range(len(tgt_seq))):
        # 获取当前时刻的缓存
        dec_cache = decoder_states[t]

        # 计算输出层梯度
        d_output = output_grads[t]
        h_dec = dec_cache[-2]  # 从缓存获取h

        # 输出层参数梯度
        grads['decoder.W_out'] += np.dot(d_output, h_dec.T)
        grads['decoder.b_out'] += d_output

        # 传递到解码器LSTM的梯度
        dh = np.dot(model.decoder.W_out.T, d_output) + dh_next
        dc = dc_next

        # LSTM反向传播
        dx, dh_prev, dc_prev, dW, db = model.decoder.lstm.backward_step(dh, dc, dec_cache)

        # 累积解码器梯度
        grads['decoder.W'] += dW
        grads['decoder.b'] += db

        # 保存梯度用于下一个时间步
        dh_next = dh_prev
        dc_next = dc_prev

    # 反向传播编码器
    dh_enc = dh_next  # 解码器初始梯度传递给编码器
    dc_enc = dc_next

    for t in reversed(range(len(src_seq))):
        x, h_enc, c_enc, enc_cache = encoder_states[t]

        # LSTM反向传播
        dx_enc, dh_prev_enc, dc_prev_enc, dW_enc, db_enc = model.encoder.lstm.backward_step(
            dh_enc, dc_enc, enc_cache)

        # 累积编码器梯度
        grads['encoder.W'] += dW_enc
        grads['encoder.b'] += db_enc

        # 更新梯度
        dh_enc = dh_prev_enc
        dc_enc = dc_prev_enc

    # 平均梯度
    seq_len = len(tgt_seq)
    grads['decoder.W_out'] /= seq_len
    grads['decoder.b_out'] /= seq_len
    grads['decoder.W'] /= seq_len
    grads['decoder.b'] /= seq_len
    grads['encoder.W'] /= len(src_seq)
    grads['encoder.b'] /= len(src_seq)

    return grads


def evaluate(model, num_samples=10):
    """评估模型"""
    references = []
    hypotheses = []

    # 生成测试数据
    test_data = generate_dummy_data(Config)[:num_samples]

    for sample in test_data:
        # 真实参考（示例数据直接使用输入）
        references.append([[str(i) for i in sample['tgt']]])

        # 模型预测
        outputs = model.forward(sample['src'], [0] * len(sample['tgt']))
        preds = [np.argmax(o) for o in outputs]
        hypotheses.append([str(p) for p in preds])

    # 计算BLEU
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu = NMTMetrics.bleu_score(ref, hyp)
        bleu_scores.append(bleu)

    return np.mean(bleu_scores)


if __name__ == "__main__":
    train()