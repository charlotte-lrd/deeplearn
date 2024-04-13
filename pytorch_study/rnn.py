import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
num_inputs, num_hiddens = 1024, 32 
num_steps = 100
batch_size = 64
num_layers = 2
gru_layer = nn.RNN(num_inputs, num_hiddens, num_layers)
X = torch.randn(num_steps, batch_size,num_inputs)
state = torch.randn(num_layers,batch_size,num_hiddens)
Y = gru_layer(X, state)
print(Y)