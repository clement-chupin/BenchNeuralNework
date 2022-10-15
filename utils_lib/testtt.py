
import torch
from torch import nn
import matplotlib.pyplot as plt
class outsider(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        # if order > 0 and in_features > 70:
        #    self.order=0

        super().__init__(in_features, (order+1)*in_features, bias=False)
        


    def get_output_size(self,):
        return (self.order+1)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(self.order,x.shape[0],)
        print(x.shape[0])
        print(self.order)
        size_a = 1/(self.order-1)

        for i in range(self.order):
            for j in range(x.shape[0]):
                if x[j] > size_a*(i-1) and x[j] < size_a*(i+1):
                    if x[j]<=size_a*(i):
                        out[i][j]=(x[j]-size_a*(i-1))/size_a
                    else:
                        out[i][j]=1+(-x[j]+size_a*(i))/size_a+torch.rand(1)*0.2

        return out

layer = outsider(124,4)

x = torch.arange(0.,1.,1./124.)
x = layer.forward(x)
print(x)
x_n = x.numpy()

plt.figure(1)

plt.plot(x_n[0])

plt.plot(x_n[1])

plt.plot(x_n[2])

plt.plot(x_n[3])
plt.show()