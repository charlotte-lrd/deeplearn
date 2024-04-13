from torch import nn
import torch

num_inputs, num_hiddens = 1024, 32 
num_steps = 100
batch_size = 64
num_layers = 2
gru_layer = nn.GRU(num_inputs, num_hiddens, num_layers)
X = torch.randn(num_steps, batch_size,num_inputs)
state = torch.randn(num_layers,batch_size,num_hiddens)
Y = gru_layer(X, state)
print(Y)
# input 和 rnn是一样的