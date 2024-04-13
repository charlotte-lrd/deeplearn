# 一个前向rnn隐藏层
# 一个反向rnn隐藏层
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
num_inputs, num_hiddens = 1024, 32 
num_steps = 100
batch_size = 64
num_layers = 2
rnn_layer = nn.RNN(num_inputs, num_hiddens, num_layers, bidirectional=True) # 词表大小 隐藏层神经元数 隐藏层层数 是否双向
X = torch.randn(num_steps, batch_size,num_inputs) # 时间步 数据集大小 词表大小
state = torch.randn(num_layers*2,batch_size,num_hiddens) #隐藏层层数*2（一个前向层一个反向层） 数据集大小 隐藏层神经元数
Y = rnn_layer(X, state)
print(Y)