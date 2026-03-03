import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import random

from simulation import simulate_fireflies_communication_range
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":
    # for interactive plots
    if platform.system() == "Darwin":
        matplotlib.use('QtAgg')
        plt.rcParams.update({'font.size': 20})
    elif platform.system() == "Linux":
        def is_headless():
            return os.environ.get("DISPLAY", "") == ""
    
    N = 100
    clock_length = 10
    T = 1000
    flash_proportion = 0.5
    path = "/Volumes/Data/other/2026_firefly_synchronization"
    # seed = np.random.randint(2**32)
    r_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
               1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    update_noise = 0.1  # to deactivate the noise set to 'None'
    
    run_params = []
    save_flash_counts = {}
    avg_num_neighbors = {}
    for r in r_range:
        save_flash_counts[r] = []
        avg_num_neighbors[r] = []
        for seed_graph in range(1):
            rng = np.random.default_rng(seed_graph)
            pos = rng.random((N, 2))
            dists = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
            communication_graph = ((dists < r) & (dists > 0)).astype(int)
            
            avg_num_neighbors[r].append(float(np.mean(np.sum(communication_graph, axis=1) / (N-1))))
            for seed in range(10000):
                rng = np.random.default_rng(seed)
                phases = rng.integers(0, clock_length, size=N)
                
                run_params.append((N, clock_length, phases, communication_graph, T, flash_proportion, r))
    
    avg_num_neighbors_df = pd.DataFrame(avg_num_neighbors)
    avg_num_neighbors_df.to_csv(f"{path}/N={N}_clock_lnegth={clock_length}_T={T}_flash_proportion={flash_proportion}_r_com_range_avg_neighbors.csv",
        index=False)
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        futures = [
            executor.submit(simulate_fireflies_communication_range, N, clock_length, phases, communication_graph, T,
                            flash_proportion, r, update_noise)
            for (N, clock_length, phases, communication_graph, T, flash_proportion, r) in run_params
        ]
    for future in as_completed(futures):
        flash_counts, phase_history, groups_history, r = future.result()
        save_flash_counts[r].append(np.max(flash_counts))
    
    save_flash_counts = pd.DataFrame(save_flash_counts)
    
    if update_noise is not None:
        noise_str = f"_update_noise={update_noise}"
    else:
        noise_str = ""
    
    save_flash_counts.to_csv(
        f"{path}/N={N}_clock_lnegth={clock_length}_T={T}_flash_proportion={flash_proportion}{noise_str}_r_com_range_flash_counts.csv",
        index=False)
    
    
