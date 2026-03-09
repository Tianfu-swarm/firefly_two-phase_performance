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

T = 10000
noise_level = 0.0  # Set None to deactivate:  0.05
Ns = [50]  # , 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200
Cs = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]  # , 14, 18, 22, 26, 30, 34, 3r8, 42, 46, 50, 70

for N in Ns:
    for C in Cs:
        path = f"/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={C}_T={T}_flash_proportion=0.5_update_noise={noise_level}_k_regular_graph_transition_flash_counts.pkl"
        
        try:
            data = pd.read_pickle(path)
            print(f"Loaded data for N={N}, C={C}")
            print(data[40].keys())
        except:
            print(f"File not found: {path}")
            continue
        
        for graph2 in [40]:
            plt.figure()
            for run in data[graph2].keys():
                plt.plot(data[graph2][run])
            plt.vlines(x=1000, ymin=0, ymax=N, colors='r', linestyles='dashed')
            plt.title(f"all-to-all -> k={graph2} | N={N}, C={C} | T={T} | noise={noise_level}")
            plt.show()
