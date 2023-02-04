
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

    {#6
        "policie"      : stable_baselines3.A2C,  #discrete, continuous action space
        "name"         : "A2C_low",
        "action_space" : [True,True],
        "compute_opti" : "auto",   #to select cpu or gpu but no exploit
        "lr_ratio"     : 0.5,
    },
    {#7
        "policie"      : stable_baselines3.DDPG,
        "name"         : "DDPG_low",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.5,
    },
    {#8
        "policie"      : stable_baselines3.DQN,
        "name"         : "DQN_low",
        "action_space" : [True,False],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.5,
    },
    {#9
        "policie"      : stable_baselines3.PPO,
        "name"         : "PPO_low",
        "action_space" : [True,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.5,
    },
    {#10
        "policie"      : stable_baselines3.SAC,
        "name"         : "SAC_low",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.5,
    },
    {#11
        "policie"      : stable_baselines3.TD3,
        "name"         : "TD3_low",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.5,
    },

    {#12
        "policie"      : stable_baselines3.A2C,  #discrete, continuous action space
        "name"         : "A2C_ulow",
        "action_space" : [True,True],
        "compute_opti" : "auto",   #to select cpu or gpu but no exploit
        "lr_ratio"     : 0.2,
    },
    {#13
        "policie"      : stable_baselines3.DDPG,
        "name"         : "DDPG_ulow",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.2,
    },
    {#14
        "policie"      : stable_baselines3.DQN,
        "name"         : "DQN_ulow",
        "action_space" : [True,False],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.2,
    },
    {#15
        "policie"      : stable_baselines3.PPO,
        "name"         : "PPO_ulow",
        "action_space" : [True,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.2,
    },
    {#16
        "policie"      : stable_baselines3.SAC,
        "name"         : "SAC_ulow",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.2,
    },
    {#117
        "policie"      : stable_baselines3.TD3,
        "name"         : "TD3_ulow",
        "action_space" : [False,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 0.2,
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
    {#1
        "policie"      : stable_baselines3.DDPG,
        "name"         : "DDPG",
        "action_space" : [False,True],
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

all_policies_discrete = [

    {#0
        "policie"      : stable_baselines3.A2C,  #discrete, continuous action space
        "name"         : "A2C",
        "action_space" : [True,True],
        "compute_opti" : "auto",   #to select cpu or gpu but no exploit
        "lr_ratio"     : 1.0,
    },
    {#3
        "policie"      : stable_baselines3.PPO,
        "name"         : "PPO",
        "action_space" : [True,True],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
    {#14
        "policie"      : stable_baselines3.DQN,
        "name"         : "DQN",
        "action_space" : [True,False],
        "compute_opti" : "auto",
        "lr_ratio"     : 1.0,
    },
]