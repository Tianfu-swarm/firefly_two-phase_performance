import numpy as np
import igraph as ig
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


data_path = f'/Volumes/Data/other/2026_firefly_synchronization/qr_f_experiments_k_graph/flash_proportion=0.5_qr_threshold=0.5_update_noise=0.0/N=150_C=50_T=1000_k_regular_graph_flash_counts.pkl'
data = pd.read_pickle(data_path)