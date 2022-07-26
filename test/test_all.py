import gym

import torch



print("Nombre de gpu : {}".format(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
print("usage :")
print(torch.cuda.mem_get_info())



from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("Ant-v4", n_envs=16)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")