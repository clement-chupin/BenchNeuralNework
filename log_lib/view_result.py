from ctypes import util
from scipy.signal import savgol_filter
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils_lib.utils import Utils


import matplotlib.pyplot as plt
import numpy as np

utils = Utils()

plt.rcParams["figure.figsize"] = (10,8)

def get_path(policie_name,env_name,fe_k,fev_l,index=0):
    global util
    #util.all_feature_extractor[]
    return os.path.join(os.path.dirname(__file__), ("../result/log_json/" +
    policie_name+ "/" +
    env_name+"/"+
    utils.all_feature_extractor[fe_k]["name"] + "_v" +
    str(utils.all_feature_extractor[fe_k]["order"][fev_l]) + "_i" +
    str(index) +".json"
    ))

def plot_one_file_by_index(
    plot_target,
    policy_i=None,
    env_j=None,
    fe_k=None,
    fe_v_k=None,
    label_plot="",
    color=None,
    marker=None,
    index=0):
    plot_one_file(
        plot_target,
        utils.all_policies[policy_i],
        utils.all_envs[env_j],
        fe_k,
        fe_v_k,
        label_plot,
        color,
        marker,
        index)

def plot_one_file(plot_target,policy=None,env=None,fe_k=None,fe_v_k=None,label_plot="",color=None,marker=None,index=0):
    path_log = get_path(
        policie_name=policy["name"],
        env_name=env["name"],
        fe_k=fe_k,
        fev_l=fe_v_k,
        index=index)
    #print(path_log)

    #path_log="../test.json"
    #print(path_log)
    data = []
    time = []
    if os.path.exists(path_log):
        print("found")
        print(path_log)

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
        # print(data)
        if not(len(data) < 40 or len(time) <40):
            ti_li = savgol_filter(time, 40, 1)
            data_li = savgol_filter(data, 40, 1)
            #plt.legend()
            plot_target.plot(ti_li,data_li,label=label_plot,c=color, marker=marker,)
    else:
        print("not_found")
        print(path_log)
    #plot_target.plot(time,data)

def index_to_tuple(index):
    return (int(index/3),index%3)
def init_plot():
    fig, axs = plt.subplots(2, 3)
    for i,policy in enumerate(utils.all_policies):
        axs[index_to_tuple(i)].set_title(policy["name"])
        #print(policy["name"])

    for ax in axs.flat:
        ax.set(xlabel='timestep', ylabel='reward')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    return (fig, axs)





def plot_env_fe_by_fe(env_j=0,index=1002):

    #fig.title("lol")
    for fev in range(len(utils.all_feature_extractor)):
        fig, axs = init_plot()
        fig.suptitle(utils.all_envs[env_j]["env"],fontweight ="bold")
        for po in range(len(utils.all_policies)):
            for fev_k in range(len(utils.all_feature_extractor[fev]["order"])):
                color = None
                marker = None
                if utils.all_feature_extractor[fev]["order"][fev_k] in [8,16,64,256]:
                    #color = "r"
                    marker="*"
                plot_one_file_by_index(
                    plot_target=axs[index_to_tuple(po)],
                    policy_i=po,
                    env_j=env_j,
                    fe_k=fev,
                    fe_v_k=fev_k,
                    label_plot=utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]),
                    color=color,
                    marker=marker,
                    index=index
                )
                print(utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]))
        # mng = plt.get_current_fig_manager()
        # mng.frame.Maximize(True)
        #plt.legend()
        plt.show()
    # for ax in axs.flat:
    #     ax.set(xlabel='timestep', ylabel='reward')
    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.label_outer()



# plot_env_fe_by_fe()

def plot_env_fe_by_fight_index(env_j=0,index=88):
    array_color = ["#0f0","#f00","#0ff","#f00","#0f0","#00f"]

    index_to_fight = [1000,1001,1002,1003]

    #fig.title("lol")
    for fev in range(len(utils.all_feature_extractor)):
        fig, axs = init_plot()
        fig.suptitle('Ant problem',fontweight ="bold")
        for iii,index_fight in enumerate(index_to_fight):
            for po in range(len(utils.all_policies)):
                for fev_k in range(len(utils.all_feature_extractor[fev]["order"])):
                    plot_one_file_by_index(
                        plot_target=axs[index_to_tuple(po)],
                        policy_i=po,
                        env_j=env_j,
                        fe_k=fev,
                        fe_v_k=fev_k,
                        label_plot=utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]),
                        color=array_color[iii],
                        index=index_fight
                    )

        plt.show()
    for ax in axs.flat:
        ax.set(xlabel='timestep', ylabel='reward')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
# for i in range(17):
#     plot_env_fe_by_fight_index(env_j=i)

# axs[0, 0].set_title('DQN')
# axs[0, 1].set_title('SAC')
# axs[1, 0].set_title('DDPG')
# axs[1, 1].set_title('PPO')






# def plot_env_fe_all_multi(env_j=0,index=1088,aaa=0,fig=None):
#     fig.suptitle(utils.all_envs[env_j]["env"],fontweight ="bold")
#     #plt.title("lol")
#     for fev in range(len(utils.all_feature_extractor)):
#         for po in [aaa]:#range(len(utils.all_policies)):
#             for fev_k in range(len(utils.all_feature_extractor[fev]["order"])):
#                 color = None
#                 marker = None
#                 if fev not in [0,29,30,31,32,33,34]:
#                 # if fev not in [0,29,30,31,32,33,34]:#,25,26,27,28,
#                     break
#                 if fev in [0]:
#                     color = "#000"
#                     marker="1"
#                 if fev in [29,30,25,]:
#                     color = "#f00"
#                 if fev in [31,32,27]:
#                     color = "#0f0"
#                 if fev in [33,34,26]:
#                     color = "#00f"
#                 if fev in [28]:
#                     color = "#f0f"

                    
#                 plot_one_file_by_index(
#                     plot_target=axs[index_to_tuple(po)],
#                     policy_i=po,
#                     env_j=env_j,
#                     fe_k=fev,
#                     fe_v_k=fev_k,
#                     label_plot=utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]),
#                     color=color,
#                     marker=marker,
#                     index=index
#                 )
#                 print(utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]))
    
# fig, axs = init_plot()


# for i in range(17):
#     fig, axs = init_plot()
#     for j in range(6):
#         print(str(i)+"___"+str(j))
#         plot_env_fe_all_multi(env_j=i,aaa=j,fig=fig)
#     #plt.legend()
#     plt.show()



def plot_env_fe_all_solo(env_j=0,index=11011,policie=0):
    #fig.suptitle(utils.all_envs[env_j]["env"],fontweight ="bold")
    plt.title(utils.all_envs[env_j]["env"])
    for fev in range(len(utils.all_feature_extractor)):
        for po in [policie]:#range(len(utils.all_policies)):
            for fev_k in range(len(utils.all_feature_extractor[fev]["order"])):
                color = None
                marker = None
                if fev not in [0,39]:
                # if fev not in [0,29,30,31,32,33,34]:#,25,26,27,28,
                    break
                if fev in [0]:
                    color = "#000"
                    marker="1"
                if fev in [29,30,25,]:
                    color = "#f00"
                if fev in [31,32,27]:
                    color = "#0f0"
                if fev in [33,34,26]:
                    color = "#00f"
                if fev in [28]:
                    color = "#f0f"

                    
                plot_one_file_by_index(
                    plot_target=plt,
                    policy_i=po,
                    env_j=env_j,
                    fe_k=fev,
                    fe_v_k=fev_k,
                    label_plot=utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]),
                    color=color,
                    marker=marker,
                    index=index
                )
                print(utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]))

for i in range(17):
    
    for j in range(12):
        plt.figure(i*12+j)
        print(str(i)+"___"+str(j))
        plot_env_fe_all_solo(env_j=i,policie=j)
        plt.legend()
        plt.show()


def plot_and_save_all(index=101):
    for env_i in len(utils.all_envs):
        for po_j in len(utils.all_policies):
            fig = plt.figure(4)
            for fe_k in len(utils.all_feature_extractor):
                for fev_l in len(utils.all_feature_extractor[fe_k]):
                    plot_one_file_by_index(
                        plot_target=fig,
                        policy_i=po_j,
                        env_j=env_i,
                        fe_k=fe_k,
                        fe_v_k=fev_l,
                        label_plot="",
                        color=None,
                        index=index
                    )
            plt.savefig("./figures/"+str(env_i)+"_"+str(po_j)+'.pdf')
            plt.show()
            plt.close()


def fake_plot(axs):
    for i in range(6):

        plot_one_file(axs[index_to_tuple(i)])





















# def plot_env_fe_by_fight(env_j=9,index=88):
#     array_color = ["#0ff","#f00","#0f0","#00f","#ff0","#f0f"]


#     all_vs_all = [

#         [1,11,21,32,42],
#         [2,12,22,33,43],
#         [3,13,23,34,44],
#         [4,14,24,35,45],
#         [5,15,25,36,46],
#         [6,16,26,37,47],
#         [7,17,27,38,48],
#         [8,18,28,39,49],
#         [9,19,29,40,50],
#         [10,20,30,41,51],
#     ]
#     cos_vs_sin = [
#         [11,21],
#         [12,22],
#         [13,23],
#         [14,24],
#         [15,25],
#         [16,26],
#         [17,27],
#         [18,28],
#         [19,29],
#         [20,30],
#     ]
#     norm_vs_oss_1 = [
#         [1,32],
#         [2,33],
#         [3,34],
#         [4,35],
#         [5,36],
#         [6,37],
#         [7,38],
#         [8,39],
#         [9,40],
#         [10,41],
#     ]
#     norm_vs_oss_2 = [
#         [21,42],
#         [22,43],
#         [23,44],
#         [24,45],
#         [25,46],
#         [26,47],
#         [27,48],
#         [28,49],
#         [29,50],
#         [30,51],
#     ]
#     array_fe = all_vs_all

#     #fig.title("lol")
#     for fev_i,fe_tab in enumerate(array_fe):
#         print(fe_tab)
#         fig, axs = init_plot()
#         fig.suptitle('Ant problem',fontweight ="bold")
#         for fi_i,fev in enumerate(fe_tab):
#             for fev_k in range(len(utils.all_feature_extractor[fev]["order"])):


#                 for po in range(len(utils.all_policies)):

#                     plot_one_file_by_index(
#                         plot_target=axs[index_to_tuple(po)],
#                         policy_i=po,
#                         env_j=env_j,
#                         fe_k=fev,
#                         fe_v_k=fev_k,
#                         #label_plot=utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]),
#                         color=array_color[fi_i],
#                         index=index
#                     )
#             # mng = plt.get_current_fig_manager()
#             # mng.frame.Maximize(True)
#         #plt.legend()
#         plt.show()


#     for ax in axs.flat:
#         ax.set(xlabel='timestep', ylabel='reward')
#     # Hide x labels and tick labels for top plots and y ticks for right plots.
#     for ax in axs.flat:
#         ax.label_outer()