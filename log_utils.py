import json
from utils import Utils
util = Utils()

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
def get_folder_path(policie_name,env_name,fe_k,index=0):
    global util
    #util.all_feature_extractor[]
    return (("./result/log_json/" + 
    policie_name+ "/" +
    env_name+"/"
    ),util.all_feature_extractor[fe_k]["name"])


def save_log_train(
    policie_name,
    env_name,
    fe_k,
    fev_l,
    timestep,
    cpu_time,
    rewards,
    fps=0,
    index=0
    ):
    #print(index)
    path_data = get_path(policie_name,env_name,fe_k,fev_l,index)
    # # data ={
    # #     "timestep" :timestep,
    # #     "cpu_time" :cpu_time,
    # #     "rewards"  :rewards,
    # #     #"index"    :index
    # #     }
    # data ={
    #     "t" :str(round(timestep, 1)), #timestep
    #     "c" :str(round(cpu_time, 5)), #cpu_time
    #     "r"  :str(round(rewards,  3)),#rewards
    #     "f"  :str(round(fps,  1)),#fps

    #     #"index"    :index
    #     }

        
    # with open(path_data, "a") as input_file:
    #     json.dump(data, input_file, sort_keys=True)
    #     input_file.write(",\n")

    t =str(round(timestep, 0))
    c = str(round(cpu_time, 5))
    r = str(round(rewards,  3))
    f = str(round(fps,  1))

    with open(path_data, "a") as input_file:
        #json.dump(data, input_file, sort_keys=True)
        input_file.write(t + "," + c + "," + r + "," + f + ",\n")