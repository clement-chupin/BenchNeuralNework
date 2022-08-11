import torch
import torch.nn as nn
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(1,1).to("cpu")
        self.flatten = torch.nn.Flatten().to("cpu")
    def forward(self, x:torch.Tensor)->torch.Tensor:
        #x = x.to("cpu")
        x = torch.reshape(x,(-1,1))
        x = self.linear_1(x)
        return x

modl = Net()
#summary(modl,(14,))

input_in = torch.tensor([[1,2,3]]).to("cpu")
print(modl.forward(input_in))
        
