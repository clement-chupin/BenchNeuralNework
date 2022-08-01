import sys
import os
  
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from utils_lib.utils import Utils 
import os
from os import path
import matplotlib.pyplot as plt
import numpy as np


u = Utils()


def get_path(policie_name,env_name,fe_k,fev_l,index=0):
    global util
    #util.all_feature_extractor[]
    return ("./result/log_json/" + 
    policie_name+ "/" +
    env_name+"/"+
    util.all_feature_extractor[fe_k]["name"] + "_v" + 
    str(util.all_feature_extractor[fe_k]["order"][fev_l]) + "_i" +
    str(index) +".json"
    )



def plot_one_file(plot_target,policy=None,env=None,fe_k=None,fe_v_k=None,index=0):
    # path_log = get_path(
    #     policie_name=policy["name"],
    #     env_name=env["name"],
    #     fe_k=fe_k,
    #     fev_l=fe_v_k,
    #     index=index)
    path_log="./test.json"

    data = []
    time = []
    with open(path_log, 'r') as fd:
        lines = fd.read().split('\n')
        for l in lines:
            split_array=l.split(",")
            if len(split_array)==5:
                #print(len(split_array))
                time.append(float(split_array[0]))
                data.append(float(split_array[2]))


    data = np.array(data)
    time = np.array(time)
    #print(data)
    plot_target.plot(time,data)


   

def plot_all(index=12):
    for env_i,env in enumerate(u.all_envs):
        for po_j,policy in enumerate(u.all_policies):
            if u.compatible_env_policie(po_j,env_i):
                for fe_k,feature in enumerate(u.all_feature_extractor):
                    for fe_v_k,fe_v in enumerate(feature["order"]):
                        plot_one_file()
                        #plt.plot()



def index_to_tuple(index):
    return (int(index/3),index%3)




# axs[0, 0].set_title('DQN')
# axs[0, 1].set_title('SAC')
# axs[1, 0].set_title('DDPG')
# axs[1, 1].set_title('PPO')
def init_plot():
    fig, axs = plt.subplots(2, 3)
    for i,policy in enumerate(u.all_policies):
        axs[index_to_tuple(i)].set_title(policy["name"])
        print(policy["name"])

    for ax in axs.flat:
        ax.set(xlabel='timestep', ylabel='reward')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    return (fig, axs)

fig, axs = init_plot()


plot_one_file(axs[0, 0])
plot_one_file(axs[0, 1])
plot_one_file(axs[1, 0])
plot_one_file(axs[1, 1])




plt.show()



