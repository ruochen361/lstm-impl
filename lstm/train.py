import numpy as np
from lstm.evaluate import evaluate, POSConfusionMatrix
from seq2seq import Seq2Seq
from losses import CrossEntropyLoss
from optimizer import Adam
import joblib


# 配置参数
class Config:
    src_vocab_size = 80
    tgt_vocab_size = 80
    hidden_size = 128
    learning_rate = 0.0001
    batch_size = 8
    epochs = 20
    max_seq_len = 20


def generate_dummy_data(config):
    # 定义20组中英文互译样本 (分词后形式)
    samples = [
        {"src": ["<sos>","Hello","<eos>"], "tgt": ["<sos>","你好","<eos>"]},
        {"src": ["<sos>","Goodbye","<eos>"], "tgt": ["<sos>","再见","<eos>"]},
        {"src": ["<sos>","Good", "morning","<eos>"], "tgt": ["<sos>","早上", "好","<eos>"]},
        {"src": ["<sos>","Good", "evening","<eos>"], "tgt": ["<sos>","晚上", "好","<eos>"]},
        {"src": ["<sos>","How", "are", "you","<eos>"], "tgt": ["<sos>","你", "好", "吗","<eos>"]},
        {"src": ["<sos>","Thank", "you","<eos>"], "tgt": ["<sos>","谢谢","<eos>"]},
        {"src": ["<sos>","You're", "welcome","<eos>"], "tgt": ["<sos>","不", "客气","<eos>"]},
        {"src": ["<sos>","I", "love", "you","<eos>"], "tgt": ["<sos>","我", "爱", "你","<eos>"]},
        {"src": ["<sos>","What's", "your", "name","<eos>"], "tgt": ["<sos>","你", "叫", "什么", "名字","<eos>"]},
        {"src": ["<sos>","My", "name", "is", "Alice","<eos>"], "tgt": ["<sos>","我", "叫", "艾丽斯","<eos>"]},
        {"src": ["<sos>","How", "old", "are", "you","<eos>"], "tgt": ["<sos>","你", "多", "大","<eos>"]},
        {"src": ["<sos>","I", "am", "20","<eos>"], "tgt": ["<sos>","我", "二十", "岁","<eos>"]},
        {"src": ["<sos>","Where", "is", "the", "bathroom","<eos>"], "tgt": ["<sos>","洗手间", "在", "哪里","<eos>"]},
        {"src": ["<sos>","This", "is", "a", "book","<eos>"], "tgt": ["<sos>","这", "是", "一本", "书","<eos>"]},
        {"src": ["<sos>","I", "like", "learning","<eos>"], "tgt": ["<sos>","我", "喜欢", "学习","<eos>"]},
        {"src": ["<sos>","What", "time", "is", "it","<eos>"], "tgt": ["<sos>","现在", "几点","<eos>"]},
        {"src": ["<sos>","Nice", "to", "meet", "you","<eos>"], "tgt": ["<sos>","很", "高兴", "认识", "你","<eos>"]},
        {"src": ["<sos>","Have", "a", "good", "day","<eos>"], "tgt": ["<sos>","祝", "你", "有", "美好", "一天","<eos>"]},
        {"src": ["<sos>","See", "you", "tomorrow","<eos>"], "tgt": ["<sos>","明天", "见","<eos>"]},
        {"src": ["<sos>","Happy", "birthday","<eos>"], "tgt": ["<sos>","生日", "快乐","<eos>"]}
    ]

    # 构建含特殊符号的词汇表
    special_symbols = ['<pad>', '<sos>', '<eos>', '<unk>']

    src_vocab = {sym: i for i, sym in enumerate(special_symbols)}
    tgt_vocab = {sym: i for i, sym in enumerate(special_symbols)}

    # 收集所有源语言词
    for sample in samples:
        for word in sample["src"]:
            if word not in src_vocab:
                src_vocab[word] = len(src_vocab)

    # 收集所有目标语言词
    for sample in samples:
        for word in sample["tgt"]:
            if word not in tgt_vocab:
                tgt_vocab[word] = len(tgt_vocab)

    # Padding处理
    max_src_len = max(len(s["src"]) for s in samples)
    max_tgt_len = max(len(s["tgt"]) for s in samples)
    # 生成训练数据
    data = []
    for sample in samples:
        # 源序列处理
        src_seq = []
        for word in sample["src"]:
            vec = np.zeros((config.src_vocab_size,1))
            vec[src_vocab.get(word, src_vocab['<unk>']),0] = 1
            src_seq.append(vec)
        # Padding填充
        while len(src_seq) < max_src_len:
            vec = np.zeros((config.src_vocab_size, 1))  # 修正为二维列向量
            vec[src_vocab['<pad>'], 0] = 1  # 二维索引赋值
            src_seq.append(vec)  # 保持所有输入为二维

        # 目标序列处理
        tgt_seq = [tgt_vocab['<sos>']] + [tgt_vocab.get(w, tgt_vocab['<unk>']) for w in sample["tgt"][1:-1]] + [
            tgt_vocab['<eos>']]
        # Padding填充
        tgt_seq = tgt_seq + [tgt_vocab['<pad>']] * (max_tgt_len - len(tgt_seq))

        data.append({'src': src_seq, 'tgt': tgt_seq})
    return data,src_vocab, tgt_vocab


def train():
    # 初始化组件
    model = Seq2Seq(Config.src_vocab_size, Config.tgt_vocab_size, Config.hidden_size)
    optimizer = Adam(lr=Config.learning_rate)
    train_data, src_vocab, tgt_vocab = generate_dummy_data(Config)
    pos_evaluator = POSConfusionMatrix()
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

        # 保存模型
        joblib.dump(model, 'model.pkl')
        # 验证评估
        avg_loss = total_loss / len(train_data)

        # 生成测试数据
        dummy_data, _, _ = generate_dummy_data(Config)
        test_data = dummy_data[:10]
        bleu = evaluate(model,test_data, src_vocab, tgt_vocab, pos_evaluator)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | BLEU: {bleu:.4f}")

        # 定期输出词性分析
        if (epoch + 1) % 5 == 0:
            print("\n词性级别分析:")
            pos_evaluator.plot()
            pos_evaluator.reset()



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

    # 前向传播编码器并获取缓存
    h_enc, c_enc, encoder_states = model.encoder.forward(src_seq)

    # 解码器前向传播保存状态
    decoder_states = []
    h_dec, c_dec = h_enc.copy(), c_enc.copy()
    for t in range(len(tgt_seq)):
        x = np.zeros((model.decoder.lstm.input_size, 1))
        output, h_dec, c_dec, dec_cache = (model.decoder.forward_step(x, h_dec, c_dec))
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

    # 梯度裁剪
    grads = clip_gradients(grads, max_norm=5.0)
    return grads


def clip_gradients(grads, max_norm):
    total_norm = 0
    for g in grads.values():
        total_norm += np.sum(g ** 2)
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for key in grads:
            grads[key] *= scale
    return grads



if __name__ == "__main__":
    train()