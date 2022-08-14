# from ast import While
# #import gym
# import sys
# import time
# import gym
# from stable_baselines3.common.utils import get_device
# from stable_baselines3 import SAC,PPO,A2C
# from stable_baselines3.common import evaluation
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
# import os
# from os import path
# import json

import sys
import os
  
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)



import numpy as np
from log_lib import log_utils
from utils_lib.utils import Utils

from time import process_time





class TrainMaster():
    def __init__(self,device="auto"):
        self.utils =    Utils()
        #self.plotter = PlotAll(utils=self.utils,lissage_coef=15)
        self.log_folder =    self.utils.log_folder
        self.model_folder =  self.utils.model_folder
        self.buffer_folder = self.utils.buffer_folder
        self.device = device
        self.all_policies =      self.utils.all_policies
        self.all_envs =          self.utils.all_envs
        self.all_feature_extractor = self.utils.all_feature_extractor

        self.log_interval =      self.utils.log_interval

        self.utils.init_folder()
    

    # def train_all_envs(self,offset_env=0,offset_policie=0,index=0): #train all
    #     for env_j in range(offset_env,len(self.all_envs)):
    #         if env_j == offset_env:
    #             self.train_env_with_all_tunning(env_j,index,offset_policie)
    #         else:
    #             self.train_env_with_all_tunning(env_j,index)
    
    
    # def train_env_with_all_tunning(self,env_j,offset_policie=0,index=0): #train anly one env but with all params

    #     for policie_i in range(offset_policie,len(self.all_policies)):
    #         #self.device = self.all_policies[policie_i]["compute_opti"]
    #         for fe_k in range(len(self.all_feature_extractor)):
    #             for fev_l in range(len(self.all_feature_extractor[fe_k]["order"])):
    #                 #print("{}_{}_{}".format(policie_i,fe_k,fev_l))
    #             #print(str(policie_i)+"__"+str(fe_k))
    #                 if self.utils.compatible_env_policie(policie_i,env_j):
    #                     print("###################################################")
    #                     print("Train of : "+self.all_policies[policie_i]["name"]+" for problem : "+self.all_envs[env_j]["name"])
    #                     self.utils.progresse_bar("policie",len(self.all_policies),policie_i)
    #                     self.utils.progresse_bar("f_extract",len(self.all_feature_extractor),fe_k)
    #                     self.utils.progresse_bar("fe_version",len(self.all_feature_extractor[fe_k]["order"]),fev_l)
    #                     self.train_and_bench(
    #                         policie_i,
    #                         env_j,
    #                         fe_k,
    #                         fev_l,
    #                         index=index
    #                         )
    
    def train_and_bench_all_fev(self,policie_i,env_j,fe_k,index=0,nb_train=None):
        for fev_l in range(len(self.utils.all_feature_extractor[fe_k]["order"])):
            self.train_and_bench(
                policie_i,
                env_j,
                fe_k,
                fev_l,
                index,
                nb_train
            )


    def train_and_bench(self,policie_i,env_j,fe_k,fev_l=0,index=0,nb_train=None):
        if not(self.utils.compatible_env_policie(policie_i,env_j)):
            return False

        policie =      self.all_policies[policie_i]["policie"]
        policie_name = self.all_policies[policie_i]["name"]
        compute_opti = self.all_policies[policie_i]["compute_opti"]
        compute_opti = self.device


        env =          self.all_envs[env_j]["env"]
        env_name =     self.all_envs[env_j]["name"]
        if nb_train is None:
            nb_train =     self.all_envs[env_j]["nb_train"]
        num_cpu  =     self.all_envs[env_j]["num_cpu"]
        

        if len(self.all_feature_extractor) <= fe_k:
            print("bad fe {}".format(fe_k))
            return False


        feature_extract = self.all_feature_extractor[fe_k]


        if len(feature_extract["order"]) <= fev_l:
            print("bad fev {}".format(fev_l))
            return False

        feature_extract_name = feature_extract["name"]
        feature_order = feature_extract["order"][fev_l]

        feature_obs_shape = feature_extract["obs_shape"]

        if policie_i in [1,5]:
            num_cpu=1

        env = self.utils.get_env(env,env_j,feature_obs_shape,num_cpu)

        policy_kwargs = self.utils.get_fe_kwargs(env,feature_extract,feature_order,compute_opti)
        if policy_kwargs is not None:
            self.timestep_i = 0
            
            model = policie(
                "MlpPolicy", env,
                policy_kwargs = policy_kwargs,
                device=compute_opti,
                verbose=1
                )
            
            self.last_time = process_time()
            self.rewards_tab = []

            def register_reward(input,_):
                self.rewards_tab.append(input['rewards'][0])
                if self.timestep_i % self.log_interval==0:
                    self.new_time = process_time()
                    log_utils.save_log_train(
                        policie_name,
                        env_name,
                        fe_k,
                        fev_l,
                        int(self.timestep_i/self.log_interval),
                        (self.new_time - self.last_time),
                        float(np.sum(self.rewards_tab)),
                        self.log_interval/(self.new_time - self.last_time),
                        index=index
                        )
                    self.rewards_tab = []
                    self.last_time = self.new_time
                self.timestep_i = self.timestep_i+1
            
            model.learn(total_timesteps=nb_train, log_interval=1000,callback = register_reward)
            del policy_kwargs
            env.close()
            del register_reward
            del model
            del env
            #del rewards_tab
        else:
            print("bad feature extractor")
            return False
        return True

    # def show_env_policie_fe(self,policie_i,env_j,fe_k,fev_l):
    #     policie =      self.all_policies[policie_i]["policie"]
    #     policie_name = self.all_policies[policie_i]["name"]

    #     env =          self.all_envs[env_j]["env"]
    #     env_name =     self.all_envs[env_j]["name"]
    #     nb_train =     self.all_envs[env_j]["nb_train"]
    #     env = self.utils.get_env(env)

    #     #feature_extract = self.all_feature_extractor[fe_k]["features_extractor"]
    #     feature_extract_name = self.all_feature_extractor[fe_k]["name"]

    #     model_path = self.utils.get_model_path(
    #         policie_name,
    #         env_name,
    #         feature_extract_name,
    #         feature_extract_var=fev_l)
    #     model = policie.load(model_path)
    #     #model = policie("MlpPolicy", env,policy_kwargs = feature_extract(device=self.device), device=self.device, verbose=1)
    #     obs = env.reset()
    #     for i in range(1000):
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, reward, done, info = env.step(action)
    #         env.render()
    #         #time.sleep(0.1)
    #         if done:
    #             obs = env.reset()
    #     env.close()

