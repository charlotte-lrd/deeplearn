from torch import nn
import torch

num_inputs, num_hiddens = 1024, 32 
num_steps = 100
batch_size = 64
num_layers = 2
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
X = torch.randn(num_steps, batch_size,num_inputs)
state = torch.randn(num_layers,batch_size,num_hiddens),torch.randn(num_layers,batch_size,num_hiddens) #一个是h一个是c
Y = lstm_layer(X, state)
print(Y)