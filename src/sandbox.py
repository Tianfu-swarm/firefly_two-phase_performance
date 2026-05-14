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


# data_path = f'/Volumes/Data/other/2026_firefly_synchronization/qr_f_experiments_k_graph/flash_proportion=0.5_qr_threshold=0.5_update_noise=0.0/N=150_C=50_T=1000_k_regular_graph_flash_counts.pkl'
# data = pd.read_pickle(data_path)


for r in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    for N in [100]:
        rng = np.random.default_rng(0)
        pos = rng.random((N, 2))
        dists = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
        communication_graph = ((dists < r) & (dists > 0)).astype(int)
        
        # Create graph from adjacency matrix
        g = ig.Graph.Adjacency(
            (communication_graph > 0).tolist(),
            mode="undirected"  # use "directed" if your graph is directed
        )
        # Check if graph is connected
        is_connected = g.is_connected()
        print(f"Connected r={r}:", is_connected)
        
        
        
        
