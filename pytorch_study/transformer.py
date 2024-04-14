import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        print(pe.shape)

        # 生成位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        print(position.shape)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        print(div_term.shape)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        print((position * div_term).shape)
        # 增加一个维度，以便在每个词向量上广播
        pe = pe.unsqueeze(0)
        print(pe.shape)
        # 注册位置编码为模型参数，但不参与梯度更新
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 将位置编码加到输入张量上
        print(x.shape)
        return x + self.pe[:, :x.size(1)]

# 测试位置编码
d_model = 512
max_len = 100
pos_encoding = PositionalEncoding(d_model, max_len)

# 创建一个输入张量，假设输入序列长度为10，词嵌入维度为512
input_tensor = torch.randn(1, 10, d_model)
output_tensor = pos_encoding(input_tensor)

print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)

# 保证每一头的输出维度相同 然后cancat 再加个无biasW 即为多头注意力

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        # 天哪 这行代码的含金量。。。。。。
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output
