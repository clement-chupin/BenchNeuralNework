import matplotlib.pyplot as plt
import json
from bench_all_final.utils_lib.utils import Utils
import log_utils
import numpy as np
import os
from os import path
from scipy.signal import savgol_filter

class PlotAll():
    def __init__(self,utils = Utils(),lissage_coef=15):
        self.utils = utils
        self.lissage_coeff = lissage_coef

        self.c_lib = [
            "#f00",
            "#0f0",
            "#00f",
            "#ff0",
            "#0ff",
            "#f0f",
            ]
    #env => def => plot by policie of the best fe
    def plot_env(self,env_j,element_to_plot="reward",index=0): #each policie his plot
        for policie_i in range(len(self.utils.all_policies)):
            plt.figure(policie_i)
            plt.title(
                self.utils.all_policies[policie_i]["name"]+
                "_"+
                self.utils.all_envs[env_j]["name"]
                )
            for fe_k in range(len(self.utils.all_feature_extractor)):
                self.plot_policie_env_fe_fev(
                    policie_i,
                    env_j,
                    fe_k,
                    fev_l=0,
                    concat_fev = True,
                    element_to_plot=element_to_plot,
                    label_plot ="auto",#self.utils.all_feature_extractor[fe_k]["name"]
                    color=None,
                    index=0
                    )
            plt.show()


    #env/fe => def => plot by policie the best fe variation
    def plot_env_fe(self,env_j,fe_k,element_to_plot="reward",index=0): #each policie his plot
        for policie_i in range(len(self.utils.all_policies)):
            plt.figure(policie_i)
            plt.title(
                self.utils.all_policies[policie_i]["name"]+
                "_"+
                self.utils.all_envs[env_j]["name"]
                )
            for fev_l in range(len(self.utils.all_feature_extractor[fe_k]["order"])):
                self.plot_policie_env_fe_fev(
                    policie_i,
                    env_j,
                    fe_k,
                    fev_l,
                    concat_fev = False,
                    element_to_plot=element_to_plot,
                    label_plot ="auto",#self.utils.all_feature_extractor[fe_k]["order"][fev_l]
                    color=None,
                    index=0
                    )
            plt.show()




    def plot_policie_env(self,policie_i,env_j,element_to_plot="r",index=0):
        plt.figure(policie_i)
        plt.title(
            self.utils.all_policies[policie_i]["name"]+
            "_"+
            self.utils.all_envs[env_j]["name"]
            )
        for fe_k in range(len(self.utils.all_feature_extractor)):
            for fev_l in range(len(self.utils.all_feature_extractor[fe_k]["order"])):
                self.plot_policie_env_fe(
                    policie_i,
                    env_j,
                    fe_k,fev_l,
                    element_to_plot,index)
        plt.show()
    def plot_policie_env_all_in_one(self,policie_i,env_j,element_to_plot="r",index=0):
        plt.figure(policie_i)
        plt.title(
            self.utils.all_policies[policie_i]["name"]+
            "_"+
            self.utils.all_envs[env_j]["name"]
            )
        for fe_k in range(len(self.utils.all_feature_extractor)):
            for fev_l in range(len(self.utils.all_feature_extractor[fe_k]["order"])):
                self.plot_policie_env_fe_fev(
                    policie_i=policie_i,
                    env_j=env_j,
                    fe_k=fe_k,
                    fev_l=fev_l,
                    concat_fev = False,
                    element_to_plot=element_to_plot,
                    label_plot ="auto",
                    color=None,
                    index=index
                    )
        plt.show()

    def plot_policie_env_all_in_one_concat(self,policie_i,env_j,element_to_plot="r",index=0):
        plt.figure(policie_i)
        plt.title(
            self.utils.all_policies[policie_i]["name"]+
            "_"+
            self.utils.all_envs[env_j]["name"]
            )
        for fe_k in range(len(self.utils.all_feature_extractor)):
            
            self.plot_policie_env_fe_fev(
                policie_i=policie_i,
                env_j=env_j,
                fe_k=fe_k,
                fev_l=0,
                concat_fev = True,
                element_to_plot=element_to_plot,
                label_plot ="auto",
                color=None,
                index=index
                )
        plt.show()
    def plot_policie_env_fe(self,policie_i,env_j,fe_k,element_to_plot="r",index=0,manual_plot=False,color=None):
        if not manual_plot:
            plt.figure(policie_i)
        plt.title(
            self.utils.all_policies[policie_i]["name"]+
            "_"+
            self.utils.all_envs[env_j]["name"]
            )
        for fev_l in range(len(self.utils.all_feature_extractor[fe_k]["order"])):
            self.plot_policie_env_fe_fev(
                policie_i,
                env_j,
                fe_k,
                fev_l,
                False,
                element_to_plot,
                label_plot="auto",
                color=color,
                index=index)
        if not manual_plot:
            plt.show()
    def plot_policie_env_fe_by_files(self,
        policie_i=0,
        env_j=0,
        fe_k=0,
        element_to_plot="r",
        label_plot ="auto",
        color=None,
        ):
        #print("lol")
        policie_name = self.utils.all_policies[policie_i]["name"]
        env_name =     self.utils.all_envs[env_j]["name"]
        (folder_path,fe_name) = log_utils.get_folder_path(policie_name,env_name,fe_k)
        dirs = os.listdir( folder_path )



        output = None
        output_tab = None
        time = None
        all_data = []

        for file in dirs:
            if (fe_name+"_") in file: 
                path_log = folder_path+file
                print(path_log)

                with open(path_log, "r") as output:
                            data = output.read().rstrip()
                            data = "["+data[:-1]+"]"
                            data = json.loads(data)
                            all_data = data

                len_max = len(all_data)
                if len_max!=0:
                    
                    data_final = np.zeros((len_max,2,),dtype=np.float64)
                    for i in range(len_max):
                                            

                        data_final[i][0] = all_data[i]["t"]
                        data_final[i][1] += float(all_data[i][element_to_plot])

                    
                    label_plot = file
                    ti_li = savgol_filter(data_final[:,0], min(51,int(len_max/2)), 3)
                    data_li = savgol_filter(data_final[:,1], min(51,int(len_max/2)), 3)
                    plt.plot(ti_li,data_li,c=color,label=label_plot)

                    plt.legend()
        plt.show()
    def plot_policie_env_fe_fev(
        self,
        policie_i=0,
        env_j=0,
        fe_k=0,
        fev_l=0,
        concat_fev = False,
        element_to_plot="r",
        label_plot ="auto",
        color=None,
        index=0
        ):
        
        if (self.utils.all_policies[policie_i]["action_space"][0] and
            self.utils.all_envs[env_j]["action_space"][0]
            or 
            self.utils.all_policies[policie_i]["action_space"][1] and
            self.utils.all_envs[env_j]["action_space"][1]
            ):

            policie_name = self.utils.all_policies[policie_i]["name"]
            env_name =     self.utils.all_envs[env_j]["name"]
            feature_name = self.utils.all_feature_extractor[fe_k]["name"]

            
            #feature_extract_var = self.utils.all_feature_extractor[fe_k]["order"][fev_l]
            output = None
            output_tab = None
            time = None
            all_data = []
            for fev_l_step in range(len(self.utils.all_feature_extractor[fe_k]["order"])):#collect all data
                if concat_fev or fev_l_step == fev_l:
                    path_log = log_utils.get_path(policie_name,env_name,fe_k,fev_l_step,index)
                    if path.exists(path_log):
                        with open(path_log, "r") as output:
                            data = output.read().rstrip()
                            data = "["+data[:-1]+"]"
                            data = json.loads(data)
                            all_data.append(data)
                    else:
                        print(path_log)
            len_max = 0
            for i in range(len(all_data)):
                if len_max < len(all_data[i]):
                    len_max = len(all_data[i])
            if len_max!=0:
                data_brut_final = np.zeros((len_max,len(all_data),2,))
                data_final = np.zeros((len_max,2,),dtype=np.float64)
                for i in range(len_max):
                    n=0
                    for j in range(len(all_data)):
                        if i < len(all_data[j]):
                            n+=1
                            data_brut_final[i][j][0] = all_data[j][i]["t"]
                            data_brut_final[i][j][1] = all_data[j][i][element_to_plot]

                            data_final[i][0] = all_data[j][i]["t"]
                            data_final[i][1] += float(all_data[j][i][element_to_plot])

                    data_final[i][1] /= n


                
                # ti_li = savgol_filter(data_final[:,0], 51, 3)
                # data_li = savgol_filter(data_final[:,1], 51, 3)
                # plt.plot(ti_li,data_li,c=color,label=label_plot)

                if label_plot == "auto":
                    label_plot = (
                        policie_name+"_"+feature_name+"_v"+
                        str(self.utils.all_feature_extractor[fe_k]["order"][fev_l])+
                        "_"+str(index)
                        )

                #plt.plot(data_final[:,1])
                #plt.plot(data_final[:,0],data_final[:,1],c=color,label=label_plot)

                
                #color = self.utils.all_feature_extractor[fe_k]["color"]
                ti_li = savgol_filter(data_final[:,0], 51, 3)
                data_li = savgol_filter(data_final[:,1], 51, 3)
                plt.plot(ti_li,data_li,c=color,label=label_plot)

                # print(data_brut_final.shape)
                # for i in range(len(all_data)):
                #     #print(i)
                #     #color = self.utils.all_feature_extractor[fe_k]["color"]
                #     ti_li = savgol_filter(data_brut_final[:,i,0], 51, 3)
                #     data_li = savgol_filter(data_brut_final[:,i,1], 51, 3)
                #     plt.plot(ti_li,data_li,c=color,label=label_plot)
               
               
                plt.legend()
            

plotter = PlotAll()

# plotter.plot_env(env_j=9,element_to_plot="r",index=42)
# plotter.plot_env(env_j=16,element_to_plot="r",index=42)

# for i in range(len(plotter.utils.all_feature_extractor)):
#     plotter.plot_env_fe(env_j=9,fe_k=i,element_to_plot="r",index=42)

#plotter.plot_policie_env_all_in_one_concat(policie_i=0,env_j=9,element_to_plot="r",index=42)


#plot_env

# plotter.plot_policie_env_fe_by_files(0,0,2)

# for po_i in range(len(plotter.utils.all_policies)):
#     plt.figure(po_i)
#     plotter.plot_policie_env_fe(policie_i=po_i,env_j=9,fe_k=3,element_to_plot="r",index=46,manual_plot=True,color="#f00")
#     plotter.plot_policie_env_fe(policie_i=po_i,env_j=9,fe_k=9,element_to_plot="r",index=46,manual_plot=True,color="#0f0")
#     plt.show()

# for po_i in range(len(plotter.utils.all_policies)):
#     plt.figure(po_i)
#     plotter.plot_policie_env_fe(policie_i=po_i,env_j=9,fe_k=5,element_to_plot="r",index=46,manual_plot=True,color="#f00")
#     plotter.plot_policie_env_fe(policie_i=po_i,env_j=9,fe_k=8,element_to_plot="r",index=46,manual_plot=True,color="#0f0")
#     plt.show()

# for po_i in range(len(plotter.utils.all_policies)):
#     plt.figure(po_i)
#     plotter.plot_policie_env_fe(policie_i=po_i,env_j=9,fe_k=7,element_to_plot="r",index=46,manual_plot=True,color="#f00")
#     plotter.plot_policie_env_fe(policie_i=po_i,env_j=9,fe_k=10,element_to_plot="r",index=46,manual_plot=True,color="#0f0")
#     plt.show()


# for fe_k in range(len(plotter.utils.all_feature_extractor)):
#     for po_i in range(len(plotter.utils.all_policies)):
    
#         plotter.plot_policie_env_fe(policie_i=po_i,env_j=9,fe_k=fe_k,element_to_plot="r",index=46)


# for fe_k in range(len(plotter.utils.all_feature_extractor)):
#     for po_i in range(len(plotter.utils.all_policies)):
#         plotter.plot_policie_env_fe(policie_i=po_i,env_j=16,fe_k=fe_k,element_to_plot="r",index=46)
