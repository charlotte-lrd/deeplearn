import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
num_inputs, num_hiddens = 1024, 32 
num_steps = 100
batch_size = 64
num_layers = 2
rnn_layer = nn.RNN(num_inputs, num_hiddens, num_layers) # 词表大小 隐藏层神经元数 隐藏层层数
X = torch.randn(num_steps, batch_size,num_inputs) # 时间步 数据集大小 词表大小
state = torch.randn(num_layers,batch_size,num_hiddens) #隐藏层层数 数据集大小 隐藏层神经元数
output, hidden = rnn_layer(X, state) 
#输出张量（output tensor）：每个时间步的隐藏状态。
#最终隐藏状态（final hidden state）：RNN 最后一个时间步的隐藏状态。
print(output)