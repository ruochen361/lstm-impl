# LSTM实现

#### 描述
本项目主要是对LSTM进行翻译器的实现。不依赖现有的神经网络框架，纯靠numpy实现。

#### 项目结构

├── activations.py        # 激活函数定义  
├── confusion_matrix.png  # 混淆矩阵  
├── decoder.py            # 解码器模块实现  
├── encoder.py            # 编码器模块实现  
├── evaluate.py           # 模型评估工具（BLEU、混淆矩阵、词性分析等）  
├── layer.py              # LSTM 层的实现（前向传播与反向传播）  
├── log.txt               # 训练日志示例  
├── losses.py             # 损失函数实现（交叉熵损失）  
├── model.pkl             # 保存训练好的模型  
├── optimizer.py          # 优化器实现  
├── seq2seq.py            # Seq2Seq 主体结构，集成编码器和解码器
└── train.py              # 训练流程主入口  


#### 训练
直接运行train.py

#### 评估
evaluate.py中是评估方法，在训练时会直接调用