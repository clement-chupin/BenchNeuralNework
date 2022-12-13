from itertools import product
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.utils import get_device
import gym


#from torchsummary import summary
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeaturesExtractor_model(BaseFeaturesExtractor):
    def __init__(self, 
    observation_space: gym.spaces.Box, 
    #features_dim: int = 256,
    FeatureExtractor = None,
    order = 3,
    device="auto"
    
    ):
        ##print(observation_space.shape)
        self.device=  device
        self.n_input= observation_space.shape[0]
        self.feature_extractor = FeatureExtractor
        self.feature_layer = FeatureExtractor["feature_layer"]
        #self.out_n_func    = FeatureExtractor["output_feature_nb"]
        self.order         = order
        self.name          = FeatureExtractor["name"]
        self.description   = FeatureExtractor["description"]

        FF_layer = self.feature_layer(in_features=self.n_input, order=self.order,device=device)

        self.n_output = FF_layer.get_output_size() #(FeatureExtractor["order"])*(n_input_channels)+1 #FLF
        # #print("in features = " +str(self.n_input))
        # #print("ff_order = "    +str(self.order ))
        # #print("out features = "+str(self.n_output))

        super(FeaturesExtractor_model, self).__init__(observation_space, self.n_output)
        self.network = nn.Sequential(
            FF_layer,
        )
        
        #summary(FF_layer)
    def forward(self, observations):
        return self.network(observations)



class NoneLayer(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.in_features = in_features
        super(NoneLayer, self).__init__()

    def get_output_size(self,):
        return self.in_features
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return x

class outsider(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_pic = 1/(self.order-1)

        self.var = 1/self.order

        
        # if order > 0 and in_features > 70:
        #    self.order=0

        super().__init__(in_features, (order)*in_features, bias=False)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)
        #print(out.shape)
        #print(x.shape)
        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic
          
          
        out = torch.min(torch.relu((out+mean)/self.var),torch.relu((mean-out)/self.var))
        
        return torch.flatten(out, start_dim=1)


class outsider2(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_pic = 1/(self.order-1)

        self.var = 2/self.order
        
        # if order > 0 and in_features > 70:
        #    self.order=0

        super().__init__(in_features, (order)*in_features, bias=False)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)
        #print(out.shape)
        #print(x.shape)
        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic
          
          
        out = torch.min(torch.relu((out+mean)/self.var),torch.relu((mean-out)/self.var))
        
        return torch.flatten(out, start_dim=1)
class outsider3(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_pic = 1/(self.order-1)

        self.var = 4/self.order
        
        # if order > 0 and in_features > 70:
        #    self.order=0

        super().__init__(in_features, (order)*in_features, bias=False)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)
        #print(out.shape)
        #print(x.shape)
        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic
          
          
        out = torch.min(torch.relu((out+mean)/self.var),torch.relu((mean-out)/self.var))
        
        return torch.flatten(out, start_dim=1)
class D_FF_LinLayer_cos(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        # if order > 0 and in_features > 70:
        #    self.order=0

        super().__init__(in_features, (order+1)**in_features, bias=False)
        c=np.array(list(product(range(order + 1), repeat=in_features)))
        with torch.no_grad():
            self.weight.copy_(torch.tensor(c, dtype=torch.float32))
        self.weight.requires_grad = False

    def get_output_size(self,):
        return (self.order+1)**self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        ##print(x.size())
        x = np.pi*super().forward(x)
        return torch.cos(x)


class D_FLF_LinLayer_cos(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features

        super().__init__(in_features, in_features*order+1, bias=False)
        c=np.zeros((in_features*order+1, in_features))
        coeff=np.arange(1, order+1)
        for i in range(in_features):
            c[1+i*order:1+order*(i+1), i]=coeff
        ##print(c)
        with torch.no_grad():
            self.weight.copy_(torch.tensor(c, dtype=torch.float32))
        self.weight.requires_grad = False

    def get_output_size(self,):
        return self.in_features*self.order+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        ##print(x)
        x = np.pi*super().forward(x)
        return torch.cos(x)


class FFP(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        super().__init__(in_features, in_features*order, bias=False)
          
        self.lamb = torch.pi*2/self.order
        self.lambda_tab = torch.arange(0,self.order,dtype=torch.float32)*self.lamb
        self.kernel = torch.stack([torch.cos(self.lambda_tab),torch.sin(self.lambda_tab)],dim=0)


    def get_output_size(self,):
        return self.in_features*self.order

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = (x*2*torch.pi)
        x = torch.stack([torch.sin(x),torch.cos(x)],dim=2)
        out =torch.matmul(x,self.kernel)
        return torch.flatten(out, start_dim=1)
class FFP11(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        super().__init__(in_features, in_features*order, bias=False)
          
        self.lamb = torch.pi/self.order
        self.lambda_tab = torch.arange(0,self.order,dtype=torch.float32)*self.lamb
        self.kernel = torch.stack([torch.cos(self.lambda_tab),torch.sin(self.lambda_tab)],dim=0)


    def get_output_size(self,):
        return self.in_features*self.order

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = (x*torch.pi)
        x = torch.stack([torch.sin(x),torch.cos(x)],dim=2)
        out =torch.matmul(x,self.kernel)
        return torch.flatten(out, start_dim=1)
class FFP12(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        super().__init__(in_features, in_features*order, bias=False)
          
        self.lamb = torch.pi/self.order
        self.lambda_tab = torch.arange(0,self.order,dtype=torch.float32)*self.lamb
        self.kernel = torch.stack([torch.cos(self.lambda_tab),torch.sin(self.lambda_tab)],dim=0)


    def get_output_size(self,):
        return self.in_features*self.order

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = (x*2*torch.pi)
        x = torch.stack([torch.sin(x),torch.cos(x)],dim=2)
        out =torch.matmul(x,self.kernel)
        return torch.flatten(out, start_dim=1)
class FFP21(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        super().__init__(in_features, in_features*order, bias=False)
          
        self.lamb = torch.pi*2/self.order
        self.lambda_tab = torch.arange(0,self.order,dtype=torch.float32)*self.lamb
        self.kernel = torch.stack([torch.cos(self.lambda_tab),torch.sin(self.lambda_tab)],dim=0)


    def get_output_size(self,):
        return self.in_features*self.order

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = (x*torch.pi)
        x = torch.stack([torch.sin(x),torch.cos(x)],dim=2)
        out =torch.matmul(x,self.kernel)
        return torch.flatten(out, start_dim=1)
class FFP152(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        super().__init__(in_features, in_features*order, bias=False)
          
        self.lamb = torch.pi*1.5/self.order
        self.lambda_tab = torch.arange(0,self.order,dtype=torch.float32)*self.lamb
        self.kernel = torch.stack([torch.cos(self.lambda_tab),torch.sin(self.lambda_tab)],dim=0)


    def get_output_size(self,):
        return self.in_features*self.order

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = (x*2*torch.pi)
        x = torch.stack([torch.sin(x),torch.cos(x)],dim=2)
        out =torch.matmul(x,self.kernel)
        return torch.flatten(out, start_dim=1)

class triangular_base(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_pic = 1/(self.order-1)

        self.var = 1/self.order


        super().__init__(in_features, self.out_size, bias=False)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)

        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic

        out = torch.min(torch.relu((out+self.size_pic)/self.var),torch.relu((self.size_pic-out)/self.var))/(self.size_pic/self.var)
                      
        return torch.flatten(out, start_dim=1)
class mix_triangular(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in range(2,self.order):
            self.layers.append(triangular_base(self.in_features,i))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class triangular_base2(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = 2 

        self.size_pic = self.var_power*self.size_ecart


        super().__init__(in_features, self.out_size, bias=False)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)

        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic

        out = torch.min(torch.relu(out+self.size_pic),torch.relu(self.size_pic-out))/(self.size_pic)
                      
        return torch.flatten(out, start_dim=1)


class mix_triangular2(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in range(2,self.order):
            self.layers.append(triangular_base2(self.in_features,i))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class triangular_base3(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = 4 
        
        self.size_pic = self.var_power*self.size_ecart


        super().__init__(in_features, self.out_size, bias=False)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)

        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic

        out = torch.min(torch.relu(out+self.size_pic),torch.relu(self.size_pic-out))/(self.size_pic)
                      
        return torch.flatten(out, start_dim=1)


class mix_triangular3(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in range(2,self.order):
            self.layers.append(triangular_base3(self.in_features,i))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class triangular_base4(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = 0.5
        
        self.size_pic = self.var_power*self.size_ecart


        super().__init__(in_features, self.out_size, bias=False)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)

        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic

        out = torch.min(torch.relu(out+self.size_pic),torch.relu(self.size_pic-out))/(self.size_pic)
                      
        return torch.flatten(out, start_dim=1)


class mix_triangular4(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in range(2,self.order):
            self.layers.append(triangular_base4(self.in_features,i))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class triangular_base5(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = 6
        
        self.size_pic = self.var_power*self.size_ecart


        super().__init__(in_features, self.out_size, bias=False)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)

        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic

        out = torch.min(torch.relu(out+self.size_pic),torch.relu(self.size_pic-out))/(self.size_pic)
                      
        return torch.flatten(out, start_dim=1)


class mix_triangular5(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in range(2,self.order):
            self.layers.append(triangular_base5(self.in_features,i))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)



class triangular_base6(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = 8
        
        self.size_pic = self.var_power*self.size_ecart


        super().__init__(in_features, self.out_size, bias=False)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)

        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic

        out = torch.min(torch.relu(out+self.size_pic),torch.relu(self.size_pic-out))/(self.size_pic)
                      
        return torch.flatten(out, start_dim=1)
class gaussian_base(nn.Linear):
    def __init__(self, in_features:int, order:int,var:float,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = var/4
        
        self.size_pic = self.size_ecart


        super().__init__(in_features, (self.order)*self.in_features, bias=True)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)

        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic

        out = torch.exp(-torch.square(out)/(0.1*self.var_power))
                      
        return torch.flatten(out, start_dim=1)


class gaussian_test_layer1(nn.Module):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        
        x = torch.exp(-torch.square(x)/0.025)


        return torch.flatten(x, start_dim=1)
class gaussian_test_layer2(nn.Module):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        
        x = torch.exp(-torch.square(x)/0.05)
                      
        return torch.flatten(x, start_dim=1)
class gaussian_test_layer3(nn.Module):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        
        x = torch.exp(-torch.square(x)/0.8)
                      
        return torch.flatten(x, start_dim=1)

class gaussian_dense_acti_base(nn.Module):
    def __init__(self, in_features:int, order:int,var,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        self.var = var
        super().__init__()

        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        
        x = torch.exp(-torch.square(x)/self.var)
                      
        return torch.flatten(x, start_dim=1)
class gaussian_test_layer1_bias(nn.Module):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        
          


    def get_output_size(self,):
        return (self.order)*self.in_features+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        
        x = torch.exp(-torch.square(x)/0.025)


        return torch.cat([torch.flatten(x, start_dim=1),torch.ones(x.size()[0],1)],dim=1)
















class gaussian_test_layer2_bias(nn.Module):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        
          


    def get_output_size(self,):
        return (self.order)*self.in_features+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        
        x = torch.exp(-torch.square(x)/0.05)
                      
        return torch.cat([torch.flatten(x, start_dim=1),torch.ones(x.size()[0],1)],dim=1)


class gaussian_test_layer3_bias(nn.Module):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        
          


    def get_output_size(self,):
        return (self.order)*self.in_features+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        
        x = torch.exp(-torch.square(x)/0.8)
                      
        return torch.cat([torch.flatten(x, start_dim=1),torch.ones(x.size()[0],1)],dim=1)

class triangle_activation_1(nn.Module):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = 1.0
        
        self.size_pic = self.var_power*self.size_ecart


        self.device = get_device(device)
        super().__init__()

        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        

    def get_output_size(self,):
        return (self.order)*self.in_features+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        x = torch.min(torch.relu(x+self.size_pic),torch.relu(self.size_pic-x))/(self.size_pic)

        return torch.cat([torch.flatten(x, start_dim=1),torch.ones(x.size()[0],1)],dim=1)
class triangle_activation_2(nn.Module):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = 2.0
        
        self.size_pic = self.var_power*self.size_ecart


        self.device = get_device(device)
        super().__init__()

        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        

    def get_output_size(self,):
        return (self.order)*self.in_features+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        x = torch.min(torch.relu(x+self.size_pic),torch.relu(self.size_pic-x))/(self.size_pic)

        return torch.cat([torch.flatten(x, start_dim=1),torch.ones(x.size()[0],1)],dim=1)
class triangle_activation_3(nn.Module):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = 4.0
        
        self.size_pic = self.var_power*self.size_ecart


        self.device = get_device(device)
        super().__init__()

        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        

    def get_output_size(self,):
        return (self.order)*self.in_features+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        x = torch.min(torch.relu(x+self.size_pic),torch.relu(self.size_pic-x))/(self.size_pic)

        return torch.cat([torch.flatten(x, start_dim=1),torch.ones(x.size()[0],1)],dim=1)

class triangle_dense_activation_base(nn.Module):
    def __init__(self, in_features:int, order:int,var,device="auto"):
        self.order = order

        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = var
        
        self.size_pic = self.var_power*self.size_ecart


        self.device = get_device(device)
        super().__init__()

        self.fc_1 = torch.nn.Linear(1,self.order).to(self.device)
        

    def get_output_size(self,):
        return (self.order)*self.in_features+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],x.size()[1],1))
        x = self.fc_1(x)
        x = torch.min(torch.relu(x+self.size_pic),torch.relu(self.size_pic-x))/(self.size_pic)

        return torch.cat([torch.flatten(x, start_dim=1),torch.ones(x.size()[0],1)],dim=1)
class triangular_base_custom(nn.Linear):
    def __init__(self, in_features:int, order:int,var:float,device="auto"):
        self.order = order
        self.in_features = in_features
        self.size_ecart = 1/(self.order-1)
        self.var_power = var
        
        self.size_pic = self.var_power*self.size_ecart


        super().__init__(in_features, (self.order)*self.in_features, bias=True)
          


    def get_output_size(self,):
        return (self.order)*self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out= torch.zeros(x.shape[0],x.shape[1],self.order)

        for i in range(self.order):
            out[:,:,i] = x-i*self.size_pic
        mean = self.size_pic

        out = torch.min(torch.relu(out+self.size_pic),torch.relu(self.size_pic-out))/(self.size_pic)
                      
        return torch.flatten(out, start_dim=1)

class mix_triangular6(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in range(2,self.order):
            self.layers.append(triangular_base6(self.in_features,i))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular7(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4,8]:
            self.layers.append(triangular_base2(self.in_features,i))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular8(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4,8,16,32]:
            self.layers.append(triangular_base2(self.in_features,i))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular9(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,5,7,9]:
            self.layers.append(triangular_base2(self.in_features,i))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_1(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,2.0))
        for i in [2,3,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))
        for i in [2,3,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,8.0))
        for i in [2,3,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,16.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_triangular_full_2(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        
        for i in [4,8]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_3(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,8.0))
        
        for i in [8,16]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_4(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,i*1.0))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_5(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        
        self.layers.append(triangular_base_custom(self.in_features,2,4.0))
        
        self.layers.append(triangular_base_custom(self.in_features,8,1.0))

        self.layers.append(triangular_base_custom(self.in_features,4,2.0))

        self.layers.append(triangular_base_custom(self.in_features,16,8.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)



class mix_triangular_full_6(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4,5,8,16]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)


class mix_triangular_full_7(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        
        for i in [2,4,8]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_8(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4,8,16]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        
        for i in [2,4,8]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_9(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4,8,16]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        
        for i in [2,4,8,12,16]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_10(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        
        for i in [2,4,8,12,16]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_11(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4,8]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        
        for i in [2,4,8,12]:
            self.layers.append(triangular_base_custom(self.in_features,i,8.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)




class mix_triangular_full_12(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4,8,12]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        for i in [2,4,8,12]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))
        for i in [2,4,8,12]:
            self.layers.append(triangular_base_custom(self.in_features,i,8.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_13(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4,8,12]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        for i in [2,4,8,12]:
            self.layers.append(triangular_base_custom(self.in_features,i,2.0))
        for i in [2,4,8,12]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_triangular_full_14(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        for i in [2,4,8]:
            self.layers.append(triangular_base_custom(self.in_features,i,2.0))
        for i in [2,4,8,12]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_triangular_full_15(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4,8,12]:
            self.layers.append(triangular_base_custom(self.in_features,i,1.0))
        for i in [2,4,8]:
            self.layers.append(triangular_base_custom(self.in_features,i,2.0))
        for i in [2,4]:
            self.layers.append(triangular_base_custom(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_1(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(gaussian_base(self.in_features,i,2.0))
        for i in [2,3,4]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))
        for i in [2,3,4]:
            self.layers.append(gaussian_base(self.in_features,i,8.0))
        for i in [2,3,4]:
            self.layers.append(gaussian_base(self.in_features,i,16.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_gaussian_full_2(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        
        for i in [4,8]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_3(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(gaussian_base(self.in_features,i,8.0))
        
        for i in [8,16]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_4(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(gaussian_base(self.in_features,i,i*1.0))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_5(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        
        self.layers.append(gaussian_base(self.in_features,2,4.0))
        
        self.layers.append(gaussian_base(self.in_features,8,1.0))

        self.layers.append(gaussian_base(self.in_features,4,2.0))

        self.layers.append(gaussian_base(self.in_features,16,8.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)



class mix_gaussian_full_6(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4,5,8,16]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_gaussian_full_7(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        
        for i in [2,4,8]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_8(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4,8,16]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        
        for i in [2,4,8]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_9(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4,8,16]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        
        for i in [2,4,8,12,16]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_10(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        
        for i in [2,4,8,12,16]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_11(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,3,4,8]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        
        for i in [2,4,8,12]:
            self.layers.append(gaussian_base(self.in_features,i,8.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)




class mix_gaussian_full_12(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4,8,12]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        for i in [2,4,8,12]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))
        for i in [2,4,8,12]:
            self.layers.append(gaussian_base(self.in_features,i,8.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_13(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4,8,12]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        for i in [2,4,8,12]:
            self.layers.append(gaussian_base(self.in_features,i,2.0))
        for i in [2,4,8,12]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_gaussian_full_14(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        for i in [2,4,8]:
            self.layers.append(gaussian_base(self.in_features,i,2.0))
        for i in [2,4,8,12]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_gaussian_full_15(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        for i in [2,4,8,12]:
            self.layers.append(gaussian_base(self.in_features,i,1.0))
        for i in [2,4,8]:
            self.layers.append(gaussian_base(self.in_features,i,2.0))
        for i in [2,4]:
            self.layers.append(gaussian_base(self.in_features,i,4.0))

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class R_FLF_Base_cos(nn.Module): ##############################################################
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        kern_array = torch.rand((1,order)).to(self.device)
        self.kern = kern_array*self.order*np.pi
    def get_output_size(self,):
        return self.in_features*self.order

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],-1,1))
        coeff_four = torch.matmul(x,self.kern)
        output = torch.cos(coeff_four)
        return output

class D_FLF_Base_cos(nn.Module): ###############################################################
    def __init__(self, in_features, order,device="auto"):

        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        kern_array = torch.arange(0,order,dtype=torch.float32).to(self.device)*np.pi
        #kern_array = torch.arange(1,order+1).to(self.device)

        self.kern = torch.reshape(kern_array,(1,-1,)).to(self.device)

    def get_output_size(self,):
        return self.in_features*self.order  

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],-1,1))#.to(self.device)
        coeff_four = torch.matmul(x,self.kern).to(self.device)
        output = torch.cos(coeff_four)
        return output

class D_FLF_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = D_FLF_Base_cos(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order   
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        return output

class R_FF_cos(nn.Module): #useless out = (1,order*2) or (2,order)
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        #device = torch.device(get_device(device))
        self.device = get_device(device)
        super().__init__()
        kern_array = torch.rand((in_features,order))*4*np.pi
        self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
        self.flatten = torch.nn.Flatten()

    def get_output_size(self,):
        return self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        coeff_four = torch.matmul(x, self.kern)
        output = torch.cos(coeff_four)

        output = self.flatten(output)
        return output

class R_FLF_cos(nn.Module): #the best
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.fourier_feature = R_FLF_Base_cos(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        return output

class L_FF_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)

        output = self.flatten(output)
        return output

class L_FF_cos_cheat(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)*0.2

        output = self.flatten(output)
        return output

class L_FF_cos_genius(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(1,order*in_features).to(self.device)
        self.sig_acti = torch.nn.Sigmoid()
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        
        weird_kern = self.linear_1(torch.tensor([1.0]))
        weird_kern = self.sig_acti(weird_kern)*8.0
        weird_kern = torch.reshape(weird_kern,(self.in_features,self.order,))


        #print(x.size())
        #print(weird_kern.size())
        output = torch.matmul(x,weird_kern)*np.pi
        #print(output.size())
        output = self.activation_1(output)

        output = self.flatten(output)
        return output

class L_FF_cos_genius_a(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(1,order*in_features).to(self.device)
        self.linear_2 = torch.nn.Linear(order*in_features,order*in_features).to(self.device)
        self.sig_acti = torch.nn.Sigmoid()
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        
        weird_kern = self.linear_1(torch.tensor([1.0]))
        weird_kern = self.linear_2(weird_kern)
        weird_kern = self.sig_acti(weird_kern)*8.0
        weird_kern = torch.reshape(weird_kern,(self.in_features,self.order,))


        #print(x.size())
        #print(weird_kern.size())
        output = torch.matmul(x,weird_kern)*np.pi
        #print(output.size())
        output = self.activation_1(output)

        output = self.flatten(output)
        return output
class L_FLF_cos_genius(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(1,order).to(self.device)
        self.sig_acti = torch.nn.Sigmoid()
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order*self.in_features 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        
        weird_kern = self.linear_1(torch.tensor([1.0]))
        weird_kern = self.sig_acti(weird_kern)*self.order
        weird_kern = torch.reshape(weird_kern,(1,self.order))


        
        x = torch.reshape(x,(x.size()[0],-1,1))
        # #print(x.size())
        # #print(weird_kern.size())


        output = torch.matmul(x,weird_kern)*np.pi
        # #print(output.size())
        output = self.activation_1(output)

        output = self.flatten(output)
        return output

class L_FLF_cos_genius_a(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(1,order).to(self.device)
        self.linear_2 = torch.nn.Linear(order,order).to(self.device)
        self.sig_acti = torch.nn.Sigmoid()
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order*self.in_features 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        
        weird_kern = self.linear_1(torch.tensor([1.0]))
        weird_kern = self.linear_2(weird_kern)
        weird_kern = self.sig_acti(weird_kern)*self.order
        weird_kern = torch.reshape(weird_kern,(1,self.order))


        
        x = torch.reshape(x,(x.size()[0],-1,1))
        # #print(x.size())
        # #print(weird_kern.size())


        output = torch.matmul(x,weird_kern)*np.pi
        # #print(output.size())
        output = self.activation_1(output)

        output = self.flatten(output)
        return output

class L_FF_cos_stupid(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order*in_features).to(self.device)
        self.sig_acti = torch.nn.Sigmoid()
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        
        weird_kern = self.linear_1(x)
        weird_kern = self.sig_acti(weird_kern)*8.0
        ##print(weird_kern.size())
        weird_kern = torch.reshape(weird_kern,(weird_kern.size()[0],self.order,self.in_features,))

        x = torch.reshape(x,(x.size()[0],-1,1))

        ##print(x.size())
        ##print(weird_kern.size())
        output = torch.matmul(weird_kern,x)*np.pi
        ##print(output.size())
        output = self.activation_1(output)

        output = self.flatten(output)
        return output

class L_FF_cos_stupid_a(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order*in_features).to(self.device)
        self.linear_2 = torch.nn.Linear(order*in_features,order*in_features).to(self.device)
        self.relu_acti = torch.nn.ReLU()
        self.sig_acti = torch.nn.Sigmoid()
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        
        weird_kern = self.linear_1(x)
        weird_kern = self.relu_acti(weird_kern)
        weird_kern = self.linear_2(weird_kern)

        weird_kern = self.sig_acti(weird_kern)*8.0
        ##print(weird_kern.size())
        weird_kern = torch.reshape(weird_kern,(weird_kern.size()[0],self.order,self.in_features,))

        x = torch.reshape(x,(x.size()[0],-1,1))

        ##print(x.size())
        ##print(weird_kern.size())
        output = torch.matmul(weird_kern,x)*np.pi
        ##print(output.size())
        output = self.activation_1(output)

        output = self.flatten(output)
        return output
class L_FLF_cos_stupid(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.sig_acti = torch.nn.Sigmoid()
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order*self.in_features 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        
        weird_kern = self.linear_1(x)
        weird_kern = self.sig_acti(weird_kern)*self.order

        weird_kern = torch.reshape(weird_kern,(weird_kern.size()[0],1,self.order,))
        x = torch.reshape(x,(x.size()[0],-1,1))
        
        ##print(x.size())
        ##print(weird_kern.size())
        output = torch.matmul(x,weird_kern)*np.pi
        ##print(output.size())
        output = self.activation_1(output)

        output = self.flatten(output)
        return output
class L_FLF_cos_stupid_a(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.linear_2 = torch.nn.Linear(order,order).to(self.device)
        self.relu_acti = torch.nn.ReLU()
        self.sig_acti = torch.nn.Sigmoid()
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order*self.in_features 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        
        weird_kern = self.linear_1(x)
        weird_kern = self.relu_acti(weird_kern)
        weird_kern = self.linear_2(weird_kern)

        weird_kern = self.sig_acti(weird_kern)*self.order

        weird_kern = torch.reshape(weird_kern,(weird_kern.size()[0],1,self.order,))
        x = torch.reshape(x,(x.size()[0],-1,1))
        
        ##print(x.size())
        ##print(weird_kern.size())
        output = torch.matmul(x,weird_kern)*np.pi
        ##print(output.size())
        output = self.activation_1(output)

        output = self.flatten(output)
        return output
class L_FF_cos_weird(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)*5

        output = self.flatten(output)
        return output
class L_FF_cos_sig(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.sigmoid = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.sigmoid(output)
        
        return output

class L_FF_cos_relu(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        
        return output
class L_FF_cos_relu_a(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.SiLU()
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        
        return output
class L_FF_cos_relu_b(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.RReLU()
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        
        return output
class L_FF_cos_relu_c(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.ReLU6()
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        
        return output
class L_FF_cos_relu_d(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.SELU()
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        
        return output

class L_FF_cos_relu_e(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.PReLU()
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        return output


class L_FF_cos_nude(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        
        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.cos
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())

        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)
        output = self.flatten(output)
        output = self.post_trait(output)

        
        return output
           
class L_FLF_Base_conv_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.conv_1 = torch.nn.ConvTranspose1d(1, order, 1,bias=False)
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation = torch.cos

    def get_output_size(self,):
        return self.in_features*self.order  
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        x = torch.reshape(x,(x.size()[0],1,-1))*np.pi
        # #print(self.order)
        # #print(x.size())
        output = self.conv_1(x)
        output = torch.permute(output,(0,2,1))

        output = self.activation(output)


        return output

class L_FLF_Base_cos(nn.Module):
    def __init__(self, in_features, order,device="auto",learning_rate=0.001):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        self.lr = learning_rate
        super().__init__()
        
        
        self.linear_1 = torch.nn.Linear(1,order).to(self.device)
        self.activation = torch.cos
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order  
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        x = torch.reshape(x,(x.size()[0],-1,1))*np.pi

        output = self.linear_1(x)
        output = self.flatten(output)

        output = self.activation(output)


        return output

class L_FLF_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        return output
class L_FLF_cos_cheat(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)*0.2
        output = self.flatten(output)
        return output

        
class L_FLF_cos_weird(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)*5
        output = self.flatten(output)
        return output

class L_FLF_cos_sig(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.sigmoid = torch.nn.Sigmoid()
        self.flatten = torch.nn.Flatten()

    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.sigmoid(output)
        
        return output
class L_FLF_cos_relu(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()

    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        
        return output

class L_FLF_cos_relu_a(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.SiLU()
        self.flatten = torch.nn.Flatten()

    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        
        return output
class L_FLF_cos_relu_b(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.RReLU()
        self.flatten = torch.nn.Flatten()

    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        
        return output

class L_FLF_cos_relu_c(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.ReLU6()
        self.flatten = torch.nn.Flatten()

    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        
        return output

class L_FLF_cos_relu_d(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.SELU()
        self.flatten = torch.nn.Flatten()

    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        return output

class L_FLF_cos_relu_e(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())
        self.relu = torch.nn.PReLU()
        self.flatten = torch.nn.Flatten()

    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        output = self.post_trait(output)
        output = self.relu(output)
        return output

           
class L_FLF_cos_nude(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)
        self.post_trait = torch.nn.Linear(self.get_output_size(),self.get_output_size())

        self.flatten = torch.nn.Flatten()

    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        output = self.post_trait(output)
        return output
class R_FLF_NNI_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        self.fourier_feature = R_FLF_Base_cos(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,order).to(self.device)

        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features*self.order

    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.flatten(output)
        return output


class D_FLF_NNI_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fourier_feature = D_FLF_Base_cos(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,order).to(self.device)
        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)

        output = self.flatten(output)
        return output


class L_FLF_NNI_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,order).to(self.device)


        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.flatten(output)
        return output

class mix_all_gaussian_01(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        # for i in [self.order]:
        #     self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(gaussian_base(self.in_features,self.order,1.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.8))#0.1  0.025
        self.layers.append(gaussian_base(self.in_features,self.order,1.0))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_all_gaussian_02(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        # for i in [self.order]:
        #     self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(gaussian_base(self.in_features,self.order,1.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.1))#0.2  0.025
        self.layers.append(gaussian_base(self.in_features,self.order,2.0))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_all_gaussian_03(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        # for i in [self.order]:
        #     self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(gaussian_base(self.in_features,self.order,1.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.025))#0.2  0.025
        self.layers.append(gaussian_base(self.in_features,self.order,4.0))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_all_gaussian_04(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        # for i in [self.order]:
        #     self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(gaussian_base(self.in_features,self.order,1.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.025))#0.2  0.025
        self.layers.append(gaussian_base(self.in_features,self.order,1.0))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_all_gaussian_05(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []
        # for i in [self.order]:
        #     self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(gaussian_base(self.in_features,self.order,1.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.8))#0.2  0.025
        self.layers.append(gaussian_base(self.in_features,self.order,4.0))


        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)





class mix_all_triangle_01(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,1.0))
        self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)



class mix_all_triangle_02(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,2.0))
        self.layers.append(triangular_base_custom(self.in_features,self.order,2.0)) 
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_all_triangle_03(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,3.0))
        self.layers.append(triangular_base_custom(self.in_features,self.order,3.0)) 
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)


class mix_all_triangle_04(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,3.0))
        self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_all_triangle_05(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,1.0))
        self.layers.append(triangular_base_custom(self.in_features,self.order,3.0)) 
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_all_triangle_06(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,1.0))
        self.layers.append(triangular_base_custom(self.in_features,self.order,2.0)) 
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)
class mix_all_triangle_07(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,2.0))
        self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)






class mix_all_weird_acti_01(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,1.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.8))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)


class mix_all_weird_acti_02(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,2.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.1))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)


class mix_all_weird_acti_03(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,3.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.025))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_all_weird_acti_04(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,1.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.025))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)


class mix_all_weird_acti_05(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangle_dense_activation_base(self.in_features,self.order,4.0))
        self.layers.append(gaussian_dense_acti_base(self.in_features,self.order,0.8))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)




class mix_all_weird_deter_01(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangular_base_custom(self.in_features,self.order,1.0))
        self.layers.append(gaussian_base(self.in_features,self.order,1.0))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)


class mix_all_weird_deter_02(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangular_base_custom(self.in_features,self.order,2.0))
        self.layers.append(gaussian_base(self.in_features,self.order,2.0))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)


class mix_all_weird_deter_03(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangular_base_custom(self.in_features,self.order,3.0))
        self.layers.append(gaussian_base(self.in_features,self.order,3.0))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)




class mix_all_weird_deter_04(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangular_base_custom(self.in_features,self.order,1.0))
        self.layers.append(gaussian_base(self.in_features,self.order,3.0))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)


class mix_all_weird_deter_05(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangular_base_custom(self.in_features,self.order,3.0))
        self.layers.append(gaussian_base(self.in_features,self.order,1.0))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_all_weird_deter_06(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangular_base_custom(self.in_features,self.order,2.0))
        self.layers.append(gaussian_base(self.in_features,self.order,1.0))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

class mix_all_weird_deter_07(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order+1
        self.in_features = in_features
        self.layers = []

        # for i in [self.order]:
        #     self.layers.append(triangle_dense_activation_base(self.in_features,self.order,0.8))#0.2  0.025
        # for i in [self.order]:
        #     self.layers.append(triangular_base_custom(self.in_features,self.order,1.0)) 2 4
        self.layers.append(triangular_base_custom(self.in_features,self.order,1.0))
        self.layers.append(gaussian_base(self.in_features,self.order,2.0))
        

        self.out_size = 0 
        for l in self.layers:
            self.out_size+=l.get_output_size()

        super().__init__(in_features, self.out_size, bias=False)
          
    def get_output_size(self,):
        return self.out_size

    def forward(self, x:torch.Tensor)->torch.Tensor:
        out = []
        for layer in self.layers:
            out.append(layer(x))
        return torch.cat(out, dim=1)

