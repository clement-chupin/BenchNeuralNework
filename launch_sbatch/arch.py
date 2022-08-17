
import stable_baselines3
import gym
all_policies = [

    {#0
        "policie"      : stable_baselines3.A2C,  #discrete, continuous action space
        "name"         : "A2C",
        "action_space" : [True,True],
        "compute_opti" : "auto",   #to select cpu or gpu but no exploit
    },
    {#1
        "policie"      : stable_baselines3.DDPG,
        "name"         : "DDPG",
        "action_space" : [False,True],
        "compute_opti" : "auto",
    },
    {#2
        "policie"      : stable_baselines3.DQN,
        "name"         : "DQN",
        "action_space" : [True,False],
        "compute_opti" : "auto",
    },
    {#3
        "policie"      : stable_baselines3.PPO,
        "name"         : "PPO",
        "action_space" : [True,True],
        "compute_opti" : "auto",
    },
    {#4
        "policie"      : stable_baselines3.SAC,
        "name"         : "SAC",
        "action_space" : [False,True],
        "compute_opti" : "auto",
    },
    {#5
        "policie"      : stable_baselines3.TD3,
        "name"         : "TD3",
        "action_space" : [False,True],
        "compute_opti" : "auto",
    },

]
env = gym.make("CartPole-v1")
env2 = gym.make("Ant-v4")

for po in all_policies:
    if po["action_space"][0] == True:

        policie = po["policie"]("MlpPolicy",env)
        print(policie)
        #print(dir(policie))


        print(policie.policy)
        #print(dir(policie.policy))

        # print(policie.policy_kwargs)
        # print(dir(policie.policy_kwargs))
        # policie_arch = policie.policy_aliases["MlpPolicy"]
        # print(policie_arch)
        # print(dir(policie_arch))

        # print(policie_arch.get_parameter())
        
    else:
        policie = po["policie"]("MlpPolicy",env2)
        print(policie)
        #print(dir(policie))


        print(policie.policy)
        #print(dir(policie.policy))

        # print(policie.policy_kwargs)
        # print(dir(policie.policy_kwargs))
        # policie_arch = policie.policy_aliases["MlpPolicy"]
        # print(policie_arch)
        # print(dir(policie_arch))

        # print(policie_arch.get_parameter())
        