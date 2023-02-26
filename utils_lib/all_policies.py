
import stable_baselines3
all_policies = [

    {#0
        "policie"      : stable_baselines3.A2C,  #discrete, continuous action space
        "name"         : "A2C",
        "action_space" : [True,True],
        "compute_opti" : "auto",   #to select cpu or gpu but no exploit
        "lr_ratio"     : 1.0,
    },
    {#1
        "policie"      : stable_baselines3.DDPG,
        "name"         : "DDPG",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
    {#2
        "policie"      : stable_baselines3.DQN,
        "name"         : "DQN",
        "action_space" : [True,False],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
    {#3
        "policie"      : stable_baselines3.PPO,
        "name"         : "PPO",
        "action_space" : [True,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
    {#4
        "policie"      : stable_baselines3.SAC,
        "name"         : "SAC",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
    {#5
        "policie"      : stable_baselines3.TD3,
        "name"         : "TD3",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },




]



all_policies_robotics = [

    {#0
        "policie"      : stable_baselines3.A2C,  #discrete, continuous action space
        "name"         : "A2C",
        "action_space" : [True,True],
        "compute_opti" : "auto",   #to select cpu or gpu but no exploit
        "lr_ratio"     : 1.0,
    },
    # {#1
    #     "policie"      : stable_baselines3.DDPG,
    #     "name"         : "DDPG",
    #     "action_space" : [False,True],
    #     "compute_opti" : "auto",
    #     "lr_ratio"     : 1.0,
    # },
    {#3
        "policie"      : stable_baselines3.PPO,
        "name"         : "PPO",
        "action_space" : [True,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
    {#4
        "policie"      : stable_baselines3.SAC,
        "name"         : "SAC",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
    {#5
        "policie"      : stable_baselines3.TD3,
        "name"         : "TD3",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
]

all_policies_discrete = [

    {#0
        "policie"      : stable_baselines3.A2C,  #discrete, continuous action space
        "name"         : "A2C",
        "action_space" : [True,True],
        "compute_opti" : "auto",   #to select cpu or gpu but no exploit
        "lr_ratio"     : 1.0,
    },
    {#2
        "policie"      : stable_baselines3.DQN,
        "name"         : "DQN",
        "action_space" : [True,False],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
    {#3
        "policie"      : stable_baselines3.PPO,
        "name"         : "PPO",
        "action_space" : [True,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
    
]