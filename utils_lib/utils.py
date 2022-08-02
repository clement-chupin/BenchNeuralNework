import torch
import torch.nn.functional as F

from stable_baselines3.common.utils import get_device
torch.device(get_device())
from stable_baselines3.common.vec_env import DummyVecEnv

import gym

import os
from os import path

import sys
sys.path.append('../utils_lib')


from utils_lib.custom_vecenv_normalize import VecNormalize
from utils_lib.feature_extractor_layers import FeaturesExtractor_model

from utils_lib.all_env import all_envs as All_Envs
# from utils_lib.all_env_norm_windows import all_input_norm as All_Norms
from utils_lib.all_feature import all_feature_extract as All_Features
from utils_lib.all_policies import all_policies as All_Policies

class Utils():
	def __init__(self):
		self.result_folder = "../result"
		self.log_folder = "../result/log_json/"
		self.model_folder = "../result/best_model/"
		self.buffer_folder = "../result/buffer_model/"
		self.num_cpu = 1
		self.log_interval = 1000

		# self.all_input_normalizer = All_Norms
		
		self.all_envs = All_Envs
		self.all_policies = All_Policies
		self.all_feature_extractor = All_Features
		

	def compatible_env_policie(self,policie_i,env_j):
		return (
			self.all_policies[policie_i]["action_space"][0] and #or = and for this case
			self.all_envs[env_j]["action_space"][0]
			or 
			self.all_policies[policie_i]["action_space"][1] and #or = and for this case
			self.all_envs[env_j]["action_space"][1]
		)	
		
			
	def get_fe_kwargs(self,env,feature_extract,feature_order,device="auto"):
		#print(feature_extract["output_feature_nb"](feature_order,env.observation_space.shape[0]))
		print(feature_extract["name"])
		print(device)
		if feature_extract["output_feature_nb"](feature_order,env.observation_space.shape[0]) > 100000:
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
		
	def get_env(self,env_name,env_i,obs_shape,num_cpu=1):
		from stable_baselines3.common.env_util import make_vec_env

		env = gym.make(env_name)
		#env = DummyVecEnv([(lambda: env)])
		env = DummyVecEnv([(lambda: env) for i in range(num_cpu)])
		env = VecNormalize(env,clip_obs=obs_shape["range"],offset_obs=obs_shape["offset"],)
		# env = VecNormalize(

		# 	env,clip_obs=obs_shape["range"],
		# 	offset_obs=obs_shape["offset"],
		# 	norm_window_defined = True,
		# 	norm_mean = self.all_input_normalizer[0][env_i][0],
		# 	norm_var = self.all_input_normalizer[0][env_i][1]
		# )
			
		
		return env

	def init_folder(self):
		def create_dir(path_dir):
			if not path.exists(path_dir):
				os.makedirs(path_dir)
	
		create_dir(self.result_folder)
		create_dir(self.log_folder)
		# create_dir(self.model_folder)
		# create_dir(self.buffer_folder)
		for p in self.all_policies:
			create_dir(self.log_folder    +p["name"])
			# create_dir(self.model_folder  +p["name"])
			# create_dir(self.buffer_folder +p["name"])
			for e in self.all_envs:
				create_dir(self.log_folder +p["name"]+"/"+e["name"])

			
