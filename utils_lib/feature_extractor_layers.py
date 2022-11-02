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
    def __init__(self, in_features:int, order:int):
        self.order = order
        self.in_features = in_features
        self.size_pic = 1/(self.order-1)

        self.var = 1/self.order
        # if order > 0 and in_features > 70:
        #    self.order=0

        super().__init__(in_features, (order+1)*in_features, bias=False)
          


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
        return out#torch.flatten(out, start_dim=1)


class outsider2(nn.Linear):
    def __init__(self, in_features:int, order:int):
        self.order = order
        self.in_features = in_features
        self.size_pic = 1/(self.order-1)

        self.var = 2/self.order
        # if order > 0 and in_features > 70:
        #    self.order=0

        super().__init__(in_features, (order+1)*in_features, bias=False)
          


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
        return out#torch.flatten(out, start_dim=1)
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

