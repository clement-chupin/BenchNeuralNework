from ctypes import util
import sys
import os
from scipy.signal import savgol_filter
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from utils_lib.utils import Utils 
import os
from os import path
import matplotlib.pyplot as plt
import numpy as np


utils = Utils()


def get_path(policie_name,env_name,fe_k,fev_l,index=0):
    global util
    #util.all_feature_extractor[]
    return ("../result/log_json/" + 
    policie_name+ "/" +
    env_name+"/"+
    utils.all_feature_extractor[fe_k]["name"] + "_v" + 
    str(utils.all_feature_extractor[fe_k]["order"][fev_l]) + "_i" +
    str(index) +".json"
    )
def plot_one_file_by_index(
    plot_target,
    policy_i=None,
    env_j=None,
    fe_k=None,
    fe_v_k=None,
    label_plot="",
    index=0):
    plot_one_file(
        plot_target,
        utils.all_policies[policy_i],
        utils.all_envs[env_j],
        fe_k,
        fe_v_k,
        label_plot,
        index)


def plot_one_file(plot_target,policy=None,env=None,fe_k=None,fe_v_k=None,label_plot="",index=0):
    path_log = get_path(
        policie_name=policy["name"],
        env_name=env["name"],
        fe_k=fe_k,
        fev_l=fe_v_k,
        index=index)

    #path_log="../test.json"

    data = []
    time = []
    if os.path.exists(path_log):
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

        ti_li = savgol_filter(time, 10, 3)
        data_li = savgol_filter(data, 10, 3)
        plt.legend()
        plot_target.plot(ti_li,data_li,label=label_plot)

    #plot_target.plot(time,data)

def index_to_tuple(index):
    return (int(index/3),index%3)
def init_plot():
    fig, axs = plt.subplots(2, 3)
    for i,policy in enumerate(utils.all_policies):
        axs[index_to_tuple(i)].set_title(policy["name"])
        print(policy["name"])

    for ax in axs.flat:
        ax.set(xlabel='timestep', ylabel='reward')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    return (fig, axs)



# def plot_all(index=12):
#     for env_i,env in enumerate(u.all_envs):
#         for po_j,policy in enumerate(u.all_policies):
#             if u.compatible_env_policie(po_j,env_i):
#                 for fe_k,feature in enumerate(u.all_feature_extractor):
#                     for fe_v_k,fe_v in enumerate(feature["order"]):
#                         plot_one_file()
#                         #plt.plot()

def plot_env_fe_by_fe(env_j=0,index=12):

    fig, axs = init_plot()
    fig.suptitle('Ant problem',fontweight ="bold")
    #fig.title("lol")
    for fev in range(len(utils.all_feature_extractor)-45):
        for po in range(len(utils.all_policies)):
            
            for fev_k in range(len(utils.all_feature_extractor[fev]["order"])):
                
                plot_one_file_by_index(
                    plot_target=axs[index_to_tuple(po)],
                    policy_i=0,
                    env_j=env_j,
                    fe_k=fev,
                    fe_v_k=fev_k,
                    label_plot=str(fev)+"_"+str(fev_k),
                    index=index
                )
    plt.show()
    for ax in axs.flat:
        ax.set(xlabel='timestep', ylabel='reward')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
                
            
plot_env_fe_by_fe()






# axs[0, 0].set_title('DQN')
# axs[0, 1].set_title('SAC')
# axs[1, 0].set_title('DDPG')
# axs[1, 1].set_title('PPO')



def fake_plot(axs):
    for i in range(6):

        plot_one_file(axs[index_to_tuple(i)])



# fig, axs = init_plot()
# fake_plot(axs)
# plt.show()



