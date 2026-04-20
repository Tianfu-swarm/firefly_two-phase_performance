import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

T_small = 10000
T_large = 100000
noise_level = 0.2  # Set None to deactivate:  0.05
Ns = [200]
Cs = [66, 70]

for N in Ns:
    for C in Cs:
        #'/Volumes/Data/other/2026_firefly_synchronization/N=50_C=14_T=10000_flash_proportion=0.5_update_noise=1.0_r_com_range_flash_counts.csv'
        path_small = f"/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={C}_T={T_small}_flash_proportion=0.5_update_noise={noise_level}_r_com_range_flash_counts.pkl"
        path_large = f"/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={C}_T={T_large}_flash_proportion=0.5_update_noise={noise_level}_r_com_range_flash_counts.pkl"
        # path_distribution = f"/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={C}_T={T}_flash_proportion=0.5_update_noise={noise_level}_r_com_range_phase_history.pkl"
        
        
        try:
            data = pd.read_pickle(path_small)
            data_2 = pd.read_pickle(path_large)
            # phase_distribution = pd.read_pickle(path_distribution)
            print(f"Loaded data for N={N}, C={C}")
        except:
            print(f"File not found: {path_small}")
            continue
        
        for r in data.keys():
            # print(phase_distribution[r])
            # plt.figure()
            for run in data[r].keys():
                if np.max(data[r][run]) < (N * 0.9) or run == 6: #
                    
                    # interesting_phase = phase_distribution[r][run]
                    # print(f"interesting run: {r} - {run}")
                    # print(interesting_phase)
                    plt.figure()
                    plt.plot(data[r][run])
                    plt.plot(data_2[r][run], alpha=0.5)
                    plt.title(f"N={N}, C={C} | T={T_small}/{T_large} | noise={noise_level} | seed={run}")
                    plt.ylim(0, N)
                    plt.show()
            # plt.vlines(x=1000, ymin=0, ymax=N, colors='r', linestyles='dashed')
            #
            
            
