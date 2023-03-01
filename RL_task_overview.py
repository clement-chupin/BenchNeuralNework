from utils_lib.utils import Utils

import random
import numpy as np


import matplotlib.pyplot as plt
utils = Utils()

env_j = 8
policie_i = 3

fe_k=173
fev_l=0

if not(utils.compatible_env_policie(policie_i,env_j)):
    print("not compatible")

policie =      utils.all_policies[policie_i]["policie"]
policie_name = utils.all_policies[policie_i]["name"]

compute_opti = utils.all_policies[policie_i]["compute_opti"]
compute_opti = "cpu"
env =          utils.all_envs[env_j]["env"]
env_name =     utils.all_envs[env_j]["name"]
if len(utils.all_feature_extractor) <= fe_k:
    print(len(utils.all_feature_extractor))
    print("bad fe {}".format(fe_k))
feature_extract = utils.all_feature_extractor[fe_k]
if len(feature_extract["order"]) <= fev_l:
    print("bad fev {}".format(fev_l))

feature_extract_name = feature_extract["name"]
feature_order = feature_extract["order"][fev_l]
feature_obs_shape = feature_extract["obs_shape"]


env = utils.get_env(env,env_j,feature_obs_shape)

policy_kwargs = utils.get_fe_kwargs(env,feature_extract,feature_order,compute_opti)
if policy_kwargs is not None:

    model = policie(
        policy="MlpPolicy",
        #learning_rate = old_lr,
        env=env,
        policy_kwargs = policy_kwargs,
        device=compute_opti,
        verbose=1,
        #seed=random.randint(100,100000),
        )



rewards_tab = []
r_epi = []

def register_reward(input,_):
    # print("---")
    # print(input)
    val =input['rewards'][0]
    r_epi.append(val)

    if input["dones"][0]:
        rewards_tab.append(np.sum(r_epi)/len(r_epi))
        r_epi.clear()
    
model.learn(
    total_timesteps=2000000, 
    log_interval=1,
    callback=register_reward
    )


plt.plot(rewards_tab)
plt.show()
obs = env.reset()
for i in range(1000000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()