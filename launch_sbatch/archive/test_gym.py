import gym 


all_envs = [

    ######### MuJoCo #######
    {#0
        "env"              :  "Ant-v4",
        "name"             :  "ant",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    {#1
        "env"              :  "HalfCheetah-v4",
        "name"             :  "halfcheetah",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    {#2
        "env"              :  "Hopper-v4",
        "name"             :  "hopper",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    {#3
        "env"              :  "HumanoidStandup-v4",
        "name"             :  "humanoidstandup",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    {#4
        "env"              :  "Humanoid-v4",
        "name"             :  "humanoid",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    {#5
        "env"              :  "InvertedDoublePendulum-v4",
        "name"             :  "inverteddoublependulum",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    {#6
        "env"              :  "InvertedPendulum-v4",
        "name"             :  "invertedpendulum",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    {#7
        "env"              :  "Reacher-v4",
        "name"             :  "reacher",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    {#8
        "env"              :  "Swimmer-v4",
        "name"             :  "swimmer",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    {#9
        "env"              :  "Walker2d-v4",
        "name"             :  "walker2d",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  8,
    },
    ######### Classic Control ########
    {#10
        "env"              :  "Acrobot-v1",
        "name"             :  "acrobot",
        "action_space"     :  [True,False],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  1,
    },
    {#11
        "env"              :  "CartPole-v1",
        "name"             :  "cartpole",
        "action_space"     :  [True,False],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  1,
    },
    {#12
        "env"              :  "MountainCarContinuous-v0",
        "name"             :  "mountaincarcontinuous",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  1,
    },
    {#13
        "env"              :  "MountainCar-v0",
        "name"             :  "moutaincar",
        "action_space"     :  [True,False],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  1,
    },
    {#14
        "env"              :  "Pendulum-v1",
        "name"             :  "pendulum",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  1,
    },
    ############## Box2d ##############
    {#15
        "env"              :  "BipedalWalker-v3",
        "name"             :  "bipedalwalker",
        "action_space"     :  [False,True],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  1,
    },
    {#16
        "env"              :  "LunarLander-v2",
        "name"             :  "lunarlander",
        "action_space"     :  [True,False],
        "nb_train"         :  1000000,
        "enable_ff"        :  True,
        "num_cpu"          :  1,
    },
]    


for env in all_envs:
    e = gym.make(env["env"])
    print(e)