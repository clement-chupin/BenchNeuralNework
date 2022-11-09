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


def get_cpu_by_multi_fev(
    policy_i=None,
    env_j=None,
    fe_k=None,
    fe_v_k=None,
    label_plot="",
    color=None,
    marker=None,
    index=0):
    a=0
    b=0
    cmpt=0
    result_all = []
    for f in range(len(fe_v_k)):
        result = get_cpu_by_index(
            policy_i,
            env_j,
            fe_k,
            f,
            label_plot,
            color,
            marker,
            index)
        if result != (0,0):
            cmpt=cmpt+1
            a+=result[0]
            b+=result[1]
            result_all.append(result)
    
    if cmpt !=0:
        result_all = np.array(result_all)
        var_percent = (np.max(result_all[:,0])-np.min(result_all[:,0]))/np.mean(result_all[:,0])
        print(var_percent)
        return (a/cmpt,b/cmpt)
    return (0,0)



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
                    time.append(float(split_array[1]))
                    data.append(float(split_array[3]))

        data = np.array(data)
        time = np.array(time)
        # plt.plot()
        # plt.show()
        return(np.mean(np.sort(data)[20:-20]),np.mean(np.sort(time)[20:-20]))

    return (0.0,0.0)
    #plot_target.plot(time,data)

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
                    name = utils.all_feature_extractor[fev]["name"]
                    print()
                    print()
                    print(out)
                    data_cpu.append(out[0])
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


def get_all_result(env_j=0,index=36):

    #fig.title("lol")
    custom_legend = []
    po_already=[]
    
    data_cpu=[]
    label_cpu=[]
    
    
    for e,en in enumerate(utils.all_envs):
        #print('\midrule\n\multirow{11}{4em}{'+en["name"].replace("_","")+'} & ',end='')
        for fev in range(len(utils.all_feature_extractor)-3):
            #print("& "+utils.all_feature_extractor[fev]["name"].replace("_",""),end=" & ")
            for po in range(len(utils.all_policies)):
                out = get_cpu_by_multi_fev(
                    policy_i=po,
                    env_j=e,
                    fe_k=fev,
                    fe_v_k=utils.all_feature_extractor[fev]["order"],
                    label_plot=utils.all_feature_extractor[fev]["name"],
                    color=None,
                    marker=None,
                    index=index
                )
                if out !=(0.0,0.0):
                    name = utils.all_feature_extractor[fev]["name"]
                    #print(name)
                    #print()
                    if po == (len(utils.all_policies)-1):
                        a=4
                        #print("{} \\\\ ".format(round(out[0],0)),end='')
                    else:
                        a=4
                        #print("{} & ".format(round(out[0],0)),end='')
                else:
                    if utils.compatible_env_policie(po,e):
                        
                        if po == (len(utils.all_policies)-1):
                            a=4
                            #print("IPP \\\\ ",end='')
                        else:
                            a=4
                            #print("IPP & ",end='')

                    else:
                        
                        if po == (len(utils.all_policies)-1):
                            a=4
                            #print("NC \\\\ ",end='')
                        else:
                            a=4
                            #print("NC & ",end='')
            #print("")


    # DFF  & 10/12 & 10/12 & 10/12 & 10/12 & 10/12 & NC    \\
    # & DFLF & 10/12 & 10/12 & 10/12 & 10/12 & 10/12 & NC    \\
    # & RFF  & 10/12 & 10/12 & 10/12 & 10/12 & 10/12 & NC    \\
    


get_all_result()




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
    

