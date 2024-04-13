from torch import nn
import torch
class RunDi(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        # 是__call__方法的实现
        output = input + 1
        return output

test = RunDi()
x = torch.tensor(1.0)
output = test(x)
print(output)