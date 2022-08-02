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
        #print(observation_space.shape)
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
        # print("in features = " +str(self.n_input))
        # print("ff_order = "    +str(self.order ))
        # print("out features = "+str(self.n_output))

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


class D_FF_LinLayer(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        # if order > 0 and in_features > 70:
        #     self.order=0

        super().__init__(in_features, (order+1)**in_features, bias=False)
        c=np.array(list(product(range(order + 1), repeat=in_features)))
        with torch.no_grad():
            self.weight.copy_(torch.tensor(c, dtype=torch.float32))
        self.weight.requires_grad = False

    def get_output_size(self,):
        return (self.order+1)**self.in_features*2

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #print(x.size())
        x = np.pi*super().forward(x)
        return torch.cat((torch.sin(x), torch.cos(x)), 1) #cos originel


class D_FLF_LinLayer(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features

        super().__init__(in_features, in_features*order+1, bias=False)
        c=np.zeros((in_features*order+1, in_features))
        coeff=np.arange(1, order+1)
        for i in range(in_features):
            c[1+i*order:1+order*(i+1), i]=coeff
        #print(c)
        with torch.no_grad():
            self.weight.copy_(torch.tensor(c, dtype=torch.float32))
        self.weight.requires_grad = False

    def get_output_size(self,):
        return (self.in_features*self.order+1)*2

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #print(x)
        x = np.pi*super().forward(x)
        return torch.cat((torch.sin(x), torch.cos(x)), 1) #cos originel

class R_FLF_Base(nn.Module): ##########--------------##################---------## 1*1 <1
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        kern_array = torch.rand((1,order)).to(self.device)
        self.kern = kern_array*self.order*np.pi
    def get_output_size(self,):
        return self.in_features*self.order*2

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],-1,1))
        coeff_four = torch.matmul(x,self.kern)
        output = torch.cat((torch.sin(coeff_four), torch.cos(coeff_four)), 1)#.numpy()
        return output

class D_FLF_Base(nn.Module): ##########--------------##################---------## 1*1 <1
    def __init__(self, in_features, order,device="auto"):

        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        kern_array = torch.arange(1,order+1,dtype=torch.float32).to(self.device)*np.pi
        #kern_array = torch.arange(1,order+1).to(self.device)

        self.kern = torch.reshape(kern_array,(1,-1,)).to(self.device)

    def get_output_size(self,):
        return self.in_features*self.order*2    

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = torch.reshape(x,(x.size()[0],-1,1))#.to(self.device)
        coeff_four = torch.matmul(x,self.kern).to(self.device)
        output = torch.cat((torch.sin(coeff_four), torch.cos(coeff_four)), 1)#.numpy()
        return output

class D_FLF(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = D_FLF_Base(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order*2   
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        return output

class R_FF(nn.Module):
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
        return self.order*2
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        coeff_four = torch.matmul(x, self.kern)
        output = torch.cat((torch.sin(coeff_four), torch.cos(coeff_four)), 1)#.numpy()

        output = self.flatten(output)
        return output

class R_FLF(nn.Module): 
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.fourier_feature = R_FLF_Base(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order*2
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        return output

class L_FF(nn.Module):#-#
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.sin
        self.activation_2 = torch.cos
        # self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        # self.linear_2 = torch.nn.Linear(4,1).to(self.device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order*2    
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output_1 = self.activation_1(output)
        output_2 = self.activation_2(output)
        output = torch.cat((output_1, output_2), 1)
        output = self.flatten(output)
        return output


class L_FLF_Base(nn.Module): #-#
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.conv_1 = torch.nn.ConvTranspose1d(1, order, 1,bias=False)
        self.activation_1 = torch.sin
        self.activation_2 = torch.cos

    def get_output_size(self,):
        return self.in_features*self.order*2   
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        x = torch.reshape(x,(x.size()[0],1,-1))*np.pi
        # print(self.order)
        # print(x.size())
        output = self.conv_1(x)
        output = torch.permute(output,(0,2,1))

        output_1 = self.activation_1(output)
        output_2 = self.activation_2(output)
        output = torch.cat((output_1, output_2), 1)
        
        return output

class L_FLF(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order*2
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        return output


class R_FLF_NNI(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        self.fourier_feature = R_FLF_Base(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,4).to(self.device)
        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features*2

    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.flatten(output)
        return output


class D_FLF_NNI(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fourier_feature = D_FLF_Base(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,4).to(self.device)
        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features*2
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.flatten(output)
        return output


class L_FLF_NNI(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fourier_feature = L_FLF_Base(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,4).to(self.device)
        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)

        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features*2
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.flatten(output)
        return output

############################## COS ################################################
############################## COS ################################################
############################## COS ################################################
############################## COS ################################################
############################## COS ################################################
############################## COS ################################################
############################## COS ################################################
############################## COS ################################################



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
        #print(x.size())
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
        #print(c)
        with torch.no_grad():
            self.weight.copy_(torch.tensor(c, dtype=torch.float32))
        self.weight.requires_grad = False

    def get_output_size(self,):
        return self.in_features*self.order+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #print(x)
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

        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)

        output = self.flatten(output)
        return output


class L_FLF_Base_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.conv_1 = torch.nn.ConvTranspose1d(1, order, 1,bias=False)
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation = torch.cos

        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
    def get_output_size(self,):
        return self.in_features*self.order  
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        x = torch.reshape(x,(x.size()[0],1,-1))*np.pi
        # print(self.order)
        # print(x.size())
        output = self.conv_1(x)
        output = torch.permute(output,(0,2,1))

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


class R_FLF_NNI_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        self.fourier_feature = R_FLF_Base_cos(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,4).to(self.device)
        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.flatten(output)
        return output


class D_FLF_NNI_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fourier_feature = D_FLF_Base_cos(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,4).to(self.device)
        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.flatten(output)
        return output


class L_FLF_NNI_cos(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fourier_feature = L_FLF_Base_cos(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,4).to(self.device)
        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)

        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.flatten(output)
        return output


############################## SIN ################################################
############################## SIN ################################################
############################## SIN ################################################
############################## SIN ################################################
############################## SIN ################################################
############################## SIN ################################################
############################## SIN ################################################
############################## SIN ################################################



class D_FF_LinLayer_sin(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features
        # if order > 0 and in_features > 70:
        #     self.order=0

        super().__init__(in_features, (order+1)**in_features, bias=False)
        c=np.array(list(product(range(order + 1), repeat=in_features)))
        with torch.no_grad():
            self.weight.copy_(torch.tensor(c, dtype=torch.float32))
        self.weight.requires_grad = False

    def get_output_size(self,):
        return (self.order+1)**self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #print(x.size())
        x = np.pi*super().forward(x)
        return torch.sin(x)


class D_FLF_LinLayer_sin(nn.Linear):
    def __init__(self, in_features:int, order:int,device="auto"):
        self.order = order
        self.in_features = in_features

        super().__init__(in_features, in_features*order+1, bias=False)
        c=np.zeros((in_features*order+1, in_features))
        coeff=np.arange(1, order+1)
        for i in range(in_features):
            c[1+i*order:1+order*(i+1), i]=coeff
        #print(c)
        with torch.no_grad():
            self.weight.copy_(torch.tensor(c, dtype=torch.float32))
        self.weight.requires_grad = False

    def get_output_size(self,):
        return self.in_features*self.order+1

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #print(x)
        x = np.pi*super().forward(x)
        return torch.sin(x)

class R_FLF_Base_sin(nn.Module): ##############################################################
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
        output = torch.sin(coeff_four)
        return output

class D_FLF_Base_sin(nn.Module): ###############################################################
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
        output = torch.sin(coeff_four)
        return output

class D_FLF_sin(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = D_FLF_Base_sin(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order   
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        return output

class R_FF_sin(nn.Module): #useless out = (1,order*2) or (2,order)
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
        output = torch.sin(coeff_four)

        output = self.flatten(output)
        return output

class R_FLF_sin(nn.Module): #the best
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.fourier_feature = R_FLF_Base_sin(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        return output

class L_FF_sin(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features

        self.device = get_device(device)
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation_1 = torch.sin

        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.order 
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.linear_1(x*np.pi)
        output = self.activation_1(output)

        output = self.flatten(output)
        return output


class L_FLF_Base_sin(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.conv_1 = torch.nn.ConvTranspose1d(1, order, 1,bias=False)
        self.linear_1 = torch.nn.Linear(in_features,order).to(self.device)
        self.activation = torch.sin

        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
    def get_output_size(self,):
        return self.in_features*self.order  
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        x = torch.reshape(x,(x.size()[0],1,-1))*np.pi
        # print(self.order)
        # print(x.size())
        output = self.conv_1(x)
        output = torch.permute(output,(0,2,1))

        output = self.activation(output)


        return output

class L_FLF_sin(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        
        self.fourier_feature = L_FLF_Base_sin(in_features, order,device)
        self.flatten = torch.nn.Flatten()
    def get_output_size(self,):
        return self.in_features*self.order
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.flatten(output)
        return output


class R_FLF_NNI_sin(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()
        self.fourier_feature = R_FLF_Base_sin(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,4).to(self.device)
        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features

    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.flatten(output)
        return output


class D_FLF_NNI_sin(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fourier_feature = D_FLF_Base_sin(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,4).to(self.device)
        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)
        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.flatten(output)
        return output


class L_FLF_NNI_sin(nn.Module):
    def __init__(self, in_features, order,device="auto"):
        self.order = order
        self.in_features = in_features
        self.device = get_device(device)
        super().__init__()

        self.fourier_feature = L_FLF_Base_sin(in_features, order,device)

        self.linear_1 = torch.nn.Linear(order,4).to(self.device)
        self.linear_1_bis = torch.nn.Linear(4,4).to(self.device)
        self.linear_2 = torch.nn.Linear(4,1).to(self.device)

        self.flatten = torch.nn.Flatten()
        #self.kern = torch.FloatTensor(np.array(kern_array)).to(self.device)
    def get_output_size(self,):
        return self.in_features
    def forward(self, x:torch.Tensor)->torch.Tensor:
         #x = x.to(self.device)
        output = self.fourier_feature(x)
        output = self.linear_1(output)
        output = self.linear_2(output)
        output = self.flatten(output)
        return output