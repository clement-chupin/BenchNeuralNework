from utils import Utils 
import os
from os import path

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

def list_all_folder(index=0):
    for env_i,env in enumerate(u.all_envs):
        for po_j,policy in enumerate(u.all_policies):
            for feature in range(len(u.all_feature_extractor)):
                for fe_v_k,fe_v in enumerate(feature["order"]):
                    if u.compatible_env_policie(po_j,env_i):
                        path_log = get_path(
                            policie_name=policy["name"],
                            env_name=env["name"],
                            fe_k=feature,
                            fev_l=fe_v_k,
                            index=index)
                        if not path.exists(path_log):
                            print(env["name"] + " " + policy["name"] + " " +str(feature) + " " + str(fe_v))



list_all_folder(index=0)