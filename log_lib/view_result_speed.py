from ctypes import util
from scipy.signal import savgol_filter
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import matplotlib.patches as mpatches
from utils_lib.utils import Utils 


import matplotlib.pyplot as plt
import numpy as np

utils = Utils()

plt.rcParams["figure.figsize"] = (10,2)

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
    
def get_cpu_by_index(
    policy_i=None,
    env_j=None,
    fe_k=None,
    fe_v_k=None,
    label_plot="",
    color=None,
    marker=None,
    index=0):
    #print(utils.all_envs[env_j])
    return get_cpu(
        utils.all_policies[policy_i],
        utils.all_envs[env_j],
        fe_k,
        fe_v_k,
        label_plot,
        color,
        marker,
        index)

def get_cpu(policy=None,env=None,fe_k=None,fe_v_k=None,label_plot="",color=None,marker=None,index=0):
    #print(env)
    path_log = get_path(
        policie_name=policy["name"],
        env_name=env["name"],
        fe_k=fe_k,
        fev_l=fe_v_k,
        index=index)


    #path_log="../test.json"
    #print(path_log)
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

        lp = int(len(data))/10
        if lp < 10:
            lp=10

        sorted_data = np.sort(data)
        min_value = np.median(sorted_data[lp:lp*2])
        max_value = np.median(data[-lp*2:])

        converge_absolute = max_value - min_value

        return converge_absolute
        offset_cut = 32.6
        len_data = 100
        filt_power = 10
        data_shifted = np.zeros((filt_power,len_data+filt_power*2))
        for i in range(filt_power):
            data_shifted[i] = np.append([np.zeros(i), data])
        
        data_sum = np.sum(a,axis=0)/filt_power

        for i in range(len_data-filt_power):
            if data_sum[i]>offset_cut:
                print(i)

        data_shift_cut = data_shifted[filt_power:-filt_power]



def index_to_tuple(index):
    return (int(index/3),index%3)




def plot_env_fe_by_fe(env_j=0,index=36):

    #fig.title("lol")
    custom_legend = []
    po_already=[]
    
    data_cpu=[]
    label_cpu=[]
    hatch = []
    color = []
    color_lib=["#f00","#f0f","#00f","#0f0","#ff0","#0ff"]
    hatch_lib=['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']#{'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
    for po in range(len(utils.all_policies)):
        print(utils.all_policies[po]["name"])
        for fev in range(len(utils.all_feature_extractor)):
            for fev_k in range(len(utils.all_feature_extractor[fev]["order"])):
                out = get_cpu_by_index(
                    policy_i=po,
                    env_j=env_j,
                    fe_k=fev,
                    fe_v_k=fev_k,
                    label_plot=utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]),
                    color=None,
                    marker=None,
                    index=index
                )
                if out !=(0.0,0.0):
                    print(utils.all_feature_extractor[fev]["name"]+"_"+str(utils.all_feature_extractor[fev]["order"][fev_k]))
                    print(out)
                    data_cpu.append(out)
                    label_cpu.append(utils.all_feature_extractor[fev]["order"][fev_k])
                    color.append(color_lib[po])
                    hatch.append(hatch_lib[fev])
                    if not(po in po_already):
                        po_already.append(po)
                        custom_legend.append(mpatches.Patch(color=color_lib[po], label=utils.all_policies[po]["name"]))

    
    


    plt.bar(
    x=np.arange(len(data_cpu)),
    height=data_cpu,
    color=color,
    tick_label=label_cpu,
    hatch=hatch
    )
    
    plt.legend(handles=custom_legend)
    plt.show()
            
plot_env_fe_by_fe()




# plot_env_fe_by_fight()



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
    

