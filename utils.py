import torch
import torch.nn.functional as F

from stable_baselines3.common.utils import get_device
torch.device(get_device())
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import gym
import stable_baselines3
import os
from os import path

import feature_extractor_layers as FeatureExtractorLayer
from feature_extractor_layers import FeaturesExtractor_model

class Utils():
    def __init__(self):
        self.log_folder = "./result/log_json/"
        self.model_folder = "./result/best_model/"
        self.buffer_folder = "./result/buffer_model/"
        self.num_cpu = 8
        self.log_interval = 100

        self.all_feature_extractor=[
            {#0
                "feature_layer"        : FeatureExtractorLayer.NoneLayer,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [0],
                "name"                 : "none",
                "description"          : "no operation\ninput => input",
                "power"                : 0,
                "color"                : None,
            },
            {#1
                "feature_layer"        : FeatureExtractorLayer.D_FF_LinLayer,
                "output_feature_nb"    : lambda order,input: ((order+1)**input),
                "order"                : [5],
                "name"                 : "dff",
                "description"          : "deterministic fourier feature, with linear layer (bad for power needed, but might be better on gpu), bad if n_input >>~ 20 \ninput => input",
                "power"                : 0,
                "color"                : None,
            },
            {#2
                "feature_layer"        : FeatureExtractorLayer.D_FLF_LinLayer,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [8], #4/8>16>>all 8
                "name"                 : "dflf_ll",
                "description"          : "deterministic fourier light feature, with linear layer (bad for power needed, but might be better on gpu) \ninput => input",
                "power"                : 0,        
                "color"                : None,    
            },
            {#3
                "feature_layer"        : FeatureExtractorLayer.D_FLF,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [8],
                "name"                 : "dflf",
                "description"          : "deterministic fourier light feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
                "power"                : 0,
                "color"                : None,
            },
            {#4
                "feature_layer"        : FeatureExtractorLayer.R_FF,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [128],#work on 4 #warnuing warning, order r_ff != order d_ff
                "name"                 : "rff",
                "description"          : "random fourier feature, with matrix layer (good for power needed, but might be worst on gpu) \ninput => input",
                "power"                : 0,
                "color"                : None,
            },
            {#5
                "feature_layer"        : FeatureExtractorLayer.R_FLF,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [8], #work on 4
                "name"                 : "rflf",
                "description"          : "random fourier light feature, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
                "power"                : 0,
                "color"                : None,
            },
            {#6
                "feature_layer"        : FeatureExtractorLayer.L_FF,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [32],#256,512],#warnuing warning, order r_ff != order d_ff
                "name"                 : "lff",
                "description"          : "learned fourier feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
                "power"                : 0,
                "color"                : None,

            },
            {#7
                "feature_layer"        : FeatureExtractorLayer.L_FLF,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [8],#,128*0],
                "name"                 : "lflf",
                "description"          : "learned fourier light feature, with matrix layer (bad for power needed, but might be better on gpu) \ninput => input",
                "power"                : 0,
                "color"                : None,
            },
            {#8
                "feature_layer"        : FeatureExtractorLayer.R_FLF_NNI,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [8], #16,32,64,128
                "name"                 : "rflfnni",
                "description"          : "random fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
                "power"                : 0,
                "color"                : "#009",
            },
            {#9
                "feature_layer"        : FeatureExtractorLayer.D_FLF_NNI,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [8],
                "name"                 : "dflfnni",
                "description"          : "deterministic fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (good for power needed, but might be worst on gpu)\ninput => input",
                "power"                : 0,
                "color"                : "#009",
            },
            {#10
                "feature_layer"        : FeatureExtractorLayer.L_FLF_NNI,
                "output_feature_nb"    : lambda order,input: (order + input),
                "order"                : [8],#work on 4
                "name"                 : "lflfnni",
                "description"          : "learned fourier light feature, with neural network who focus on each input before global computaion, with matrix layer (bad for power needed, but might be better on gpu)\ninput => input",
                "power"                : 0,
                "color"                : "#009",
            },
        ]

        self.all_policies=[#HER to add

            {#0
                "policie"      : stable_baselines3.A2C,  #disc, conti
                "name"         : "A2C",
                "action_space" : [True,True],
                "compute_opti" : "cpu",
                "memory_opti"  : False,
            },
            {#1
                "policie"      : stable_baselines3.DDPG,
                "name"         : "DDPG",
                "action_space" : [False,True],
                "compute_opti" : "auto",
                "memory_opti"  : True,
            },
            {#2
                "policie"      : stable_baselines3.DQN,
                "name"         : "DQN",
                "action_space" : [True,False],
                "compute_opti" : "auto",
                "memory_opti"  : True,
            },
            {#3
                "policie"      : stable_baselines3.PPO,
                "name"         : "PPO",
                "action_space" : [True,True],
                "compute_opti" : "auto",
                "memory_opti"  : False,
            },
            {#4
                "policie"      : stable_baselines3.SAC,
                "name"         : "SAC",
                "action_space" : [False,True],
                "compute_opti" : "auto",
                "memory_opti"  : True,
            },
            {#5
                "policie"      : stable_baselines3.TD3,
                "name"         : "TD3",
                "action_space" : [False,True],
                "compute_opti" : "auto",
                "memory_opti"  : True,
            },

        ]


        self.all_envs = [
            ######### MuJoCo #######
            {#0
                "env"              :  "Ant-v4",
                "name"             :  "ant",
                "action_space"     :  [False,True],
                "nb_train"         :  1000000,
                "enable_ff"        :  True,
                "num_cpu"          :  2,
            },
            {#1
                "env"              :  "HalfCheetah-v4",
                "name"             :  "halfcheetah",
                "action_space"     :  [False,True],
                "nb_train"         :  1000000,
                "enable_ff"        :  True,
                "num_cpu"          :  4,
            },
            {#2
                "env"              :  "Hopper-v4",
                "name"             :  "hopper",
                "action_space"     :  [False,True],
                "nb_train"         :  1000000,
                "enable_ff"        :  True,
                "num_cpu"          :  4,
            },
            {#3
                "env"              :  "HumanoidStandup-v4",
                "name"             :  "humanoidstandup",
                "action_space"     :  [False,True],
                "nb_train"         :  1000000,
                "enable_ff"        :  True,
                "num_cpu"          :  1,
            },
            {#4
                "env"              :  "Humanoid-v4",
                "name"             :  "humanoid",
                "action_space"     :  [False,True],
                "nb_train"         :  1000000,
                "enable_ff"        :  True,
                "num_cpu"          :  1,
            },
            {#5
                "env"              :  "InvertedDoublePendulum-v4",
                "name"             :  "inverteddoublependulum",
                "action_space"     :  [False,True],
                "nb_train"         :  500000,
                "enable_ff"        :  True,
                "num_cpu"          :  4,
            },
            {#6
                "env"              :  "InvertedPendulum-v4",
                "name"             :  "invertedpendulum",
                "action_space"     :  [False,True],
                "nb_train"         :  500000,
                "enable_ff"        :  True,
                "num_cpu"          :  4,
            },
            {#7
                "env"              :  "Reacher-v4",
                "name"             :  "reacher",
                "action_space"     :  [False,True],
                "nb_train"         :  500000,
                "enable_ff"        :  True,
                "num_cpu"          :  4,
            },
            {#8
                "env"              :  "Swimmer-v4",
                "name"             :  "swimmer",
                "action_space"     :  [False,True],
                "nb_train"         :  1000000,
                "enable_ff"        :  True,
                "num_cpu"          :  4,
            },
            {#9
                "env"              :  "Walker2d-v4",
                "name"             :  "walker2d",
                "action_space"     :  [False,True],
                "nb_train"         :  1000000,
                "enable_ff"        :  True,
                "num_cpu"          :  4,
            },
            ######### Classic Control ########
            {#10
                "env"              :  "Acrobot-v1",
                "name"             :  "acrobot",
                "action_space"     :  [True,False],
                "nb_train"         :  1000000,
                "enable_ff"        :  True,
                "num_cpu"          :  8,
            },
            {#11
                "env"              :  "CartPole-v1",
                "name"             :  "cartpole",
                "action_space"     :  [True,False],
                "nb_train"         :  500000,
                "enable_ff"        :  True,
                "num_cpu"          :  8,
            },
            {#12
                "env"              :  "MountainCarContinuous-v0",
                "name"             :  "mountaincarcontinuous",
                "action_space"     :  [False,True],
                "nb_train"         :  500000,
                "enable_ff"        :  True,
                "num_cpu"          :  8,
            },
            {#13
                "env"              :  "MountainCar-v0",
                "name"             :  "moutaincar",
                "action_space"     :  [True,False],
                "nb_train"         :  500000,
                "enable_ff"        :  True,
                "num_cpu"          :  8,
            },
            {#14
                "env"              :  "Pendulum-v1",
                "name"             :  "pendulum",
                "action_space"     :  [False,True],
                "nb_train"         :  500000,
                "enable_ff"        :  True,
                "num_cpu"          :  8,
            },
            ############## Box2d ##############
            {#15
                "env"              :  "BipedalWalker-v3",
                "name"             :  "bipedalwalker",
                "action_space"     :  [False,True],
                "nb_train"         :  1000000,
                "enable_ff"        :  True,
                "num_cpu"          :  8,
            },
            {#16
                "env"              :  "LunarLander-v2",
                "name"             :  "lunarlander",
                "action_space"     :  [True,False],
                "nb_train"         :  500000,
                "enable_ff"        :  True,
                "num_cpu"          :  8,
            },
        ]    
        
    def get_fe_kwargs(self,env,feature_extract,feature_order,device="auto"):
        #print(feature_extract["output_feature_nb"](feature_order,env.observation_space.shape[0]))
        print(feature_extract["name"])
        print(device)
        if feature_extract["output_feature_nb"](feature_order,env.observation_space.shape[0]) > 10000000:
            print("unable to bench : "+feature_extract["name"] +" for : "+str(env))
            print(feature_extract["output_feature_nb"](feature_order,env.observation_space.shape[0]))
            return None
        
        return dict(
            features_extractor_class=FeaturesExtractor_model,
            features_extractor_kwargs=dict(
                #observation_space = env.observation_space,
                FeatureExtractor = feature_extract,
                order = feature_order,
                device = device
            ),
        )
    def progresse_bar(self,name,max_size, size_now):
        log_str = str(size_now)+"/"+str(max_size)
        while max_size < 15:
            (max_size, size_now) = (max_size*2, size_now*2)
        
        
        offset_x = 5
        print("#"*(offset_x)+" "+name+" "+"#"*(max_size-offset_x-len(name)))
        print("#"+"-"*size_now+" "*(max_size-size_now)+"#")
        print("#"*(offset_x)+" "+log_str+" "+"#"*(max_size-offset_x-len(log_str)))

    def get_model_path(self,policie_name,env_name,feature_extract_name,feature_extract_var=0):
        #feature_extract_name + "_v" + 
        #str(feature_extract_var)
        
        return (
            self.model_folder+
            policie_name+ "/" +
            env_name+"_"+feature_extract_name + "_v" + 
            str(feature_extract_var)
        )
        
    def get_env(self,env_name,num_cpu=1):
        from stable_baselines3.common.env_util import make_vec_env

        env = gym.make(env_name)
        #env = DummyVecEnv([(lambda: env)])
        env = DummyVecEnv([(lambda: env) for i in range(num_cpu)])
        env = VecNormalize(env)
        return env

    def init_folder(self):
        def create_dir(path_dir):
            if not path.exists(path_dir):
                os.makedirs(path_dir)
    
        create_dir("result")
        create_dir(self.log_folder)
        create_dir(self.model_folder)
        create_dir(self.buffer_folder)
        for p in self.all_policies:
            create_dir(self.log_folder    +p["name"])
            create_dir(self.model_folder  +p["name"])
            create_dir(self.buffer_folder +p["name"])
            for e in self.all_envs:
                create_dir(self.log_folder +p["name"]+"/"+e["name"])

            
