import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def cast_to_python_types_and_normalize(d):
    new_dict = {}
    
    for key, value in d.items():
        # convert tuple key to pure Python ints
        new_key = tuple(int(x) for x in key)
        
        # convert nested dictionary
        new_value = {
            "counter": int(value["counter"]),
            "distribution": value["distribution"] / value["counter"]
        }
        
        new_dict[new_key] = new_value
    
    return new_dict

C = 10

with open(f'/Volumes/Data/other/2026_firefly_synchronization/N=100_clock_lnegth={C}_T=1000_flash_proportion=0.5_k_regular_graph_init_state_failed.pkl', 'rb') as f:
    data_failed = pickle.load(f)
data_failed = data_failed[100]
    
with open(f'/Volumes/Data/other/2026_firefly_synchronization/N=100_clock_lnegth={C}_T=1000_flash_proportion=0.5_k_regular_graph_init_state_sucess.pkl', 'rb') as f:
    data_success = pickle.load(f)
data_success = data_success[100]

print(data_success[0])

for d in data_failed:
    plt.hist(d, bins=100)
    plt.show()