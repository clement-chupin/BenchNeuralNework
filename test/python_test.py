import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)

import torch 
import gym
print(torch.__version__)
print(gym.__version__)



with open("log_out.txt","a") as f:
    f.write("\nend")
