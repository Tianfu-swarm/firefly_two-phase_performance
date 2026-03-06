import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


C = 10
N = 100
T = 1000
t_switch = 100
flash_proportion = 0.5
noise_level = 0.0
with open(f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_clock_lnegth={C}_T={T}_flash_proportion={flash_proportion}_update_noise={noise_level}_k_regular_graph_transition_flash_counts.pkl', 'rb') as f:
    flash_count = pickle.load(f)
print(flash_count[90].keys())

print(flash_count[90][1])
for i in range(100):
    plt.plot(flash_count[90][i])
    plt.vlines(t_switch, 0, max(flash_count[90][i]), colors='r', linestyles='dashed')
    plt.xlabel("time")
    plt.ylabel("number of flashes")
    plt.show()