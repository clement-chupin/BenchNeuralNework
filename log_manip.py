from utils import Utils 
import os
from os import path

u = Utils()

util = u
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
    cmpt=0
    with open("run_correct.sh", "a") as input_file:
        for env_i,env in enumerate(u.all_envs):
            for po_j,policy in enumerate(u.all_policies):
                for fe_k,feature in enumerate(u.all_feature_extractor):
                    for fe_v_k,fe_v in enumerate(feature["order"]):
                        if u.compatible_env_policie(po_j,env_i):
                            path_log = get_path(
                                policie_name=policy["name"],
                                env_name=env["name"],
                                fe_k=fe_k,
                                fev_l=fe_v_k,
                                index=index)
                            if not path.exists(path_log):
                                cmpt=cmpt+1
                                #print(path_log)
                                input_file.write("\nsbatch -o './log_meso/log_propre' single_run.sh "+str(env_i) + " " +str(po_j) + " " +str(fe_k))
                            else:
                                
                                with open(path_log, 'r') as fp:
                                    for count, line in enumerate(fp):
                                        pass
                                if count not in range(490,510) and count not in range(990,1010):
                                    os.remove(path_log)


                                if False:
                                    with open(path_log, 'r+') as fp:
                                        lines = fp.readlines()
                                        fp.seek(0)
                                        fp.truncate()
                                        fp.writelines(lines[1:])
                                if False:
                                    print("nop")
                                    

    print(cmpt)


list_all_folder(index=12)