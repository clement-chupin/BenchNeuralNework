
import torch
from torch import nn
import matplotlib.pyplot as plt
class outsider(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_pic = 1/(self.order-1)
        # if order > 0 and in_features > 70:
        #    self.order=0

        super().__init__(in_features, (order+1)*in_features, bias=False)
        


    def get_output_size(self,):
        return (self.order+1)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order+1)
        zeros = torch.zeros(x.shape[1])

        for i in range(x.shape[0]):#batch_size
            for j in range(self.order+1):
                x_pic = x[i]
                mean = self.size_pic*j*(self.order-1)
                var = 1/self.order
                x_pic = x_pic/var
                out[i,:,j] = (
                    torch.relu(x_pic-mean+1) * torch.heaviside(1-x_pic+mean-1,zeros) + 
                    torch.relu(2-x_pic+mean-1) * torch.heaviside(x_pic-mean+0.000000001,zeros)
                    )

        
        return torch.flatten(out, start_dim=1)








