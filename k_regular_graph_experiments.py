import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import random
import pickle
from tqdm import tqdm

from simulation import simulate_fireflies_k_regular_graph
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
    k_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90,
               100]  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90,
    update_noise = 0.2  # to deactivate the noise set to 'None'
    
    run_params = []
    save_flash_counts = {}
    save_phase_history = {}
    save_init_state_failed = {}
    save_init_state_success = {}
    avg_num_neighbors = {}
    for k in k_range:
        save_flash_counts[k] = np.zeros(10000)
        avg_num_neighbors[k] = []
        save_phase_history[k] = np.zeros((10000, N))
        save_init_state_failed[k] = np.zeros((10000, N))
        save_init_state_success[k] = np.zeros((10000, N))
        for seed_graph in range(1):
            if k == N:
                communication_graph = np.ones((N, N))
                np.fill_diagonal(communication_graph, 0)
            else:
                random.seed(seed_graph)
                np.random.seed(seed_graph)
                ig.set_random_number_generator(random)
                G = ig.Graph.K_Regular(N, k)
                communication_graph = np.array(G.get_adjacency().data)
            
            avg_num_neighbors[k].append(float(np.mean(np.sum(communication_graph, axis=1) / (N - 1))))
            for seed in range(10000):
                rng = np.random.default_rng(seed)
                phases = rng.integers(0, clock_length, size=N)
                
                run_params.append((N, clock_length, phases, communication_graph, T, flash_proportion, k, seed))
    
    avg_num_neighbors_df = pd.DataFrame(avg_num_neighbors)
    avg_num_neighbors_df.to_csv(
        f"{path}/N={N}_clock_lnegth={clock_length}_T={T}_flash_proportion={flash_proportion}_k_regular_graph_avg_neighbors.csv",
        index=False)
    print(f"done setting up the parameters ...")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        futures = [
            executor.submit(simulate_fireflies_k_regular_graph, N, clock_length, phases, communication_graph, T,
                            flash_proportion, k, seed, update_noise)
            for (N, clock_length, phases, communication_graph, T, flash_proportion, k, seed) in run_params
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            flash_counts, phase_history, groups_history, k, init_clock_state, seed = future.result()
            save_flash_counts[k][seed] = np.max(flash_counts)
            if np.max(flash_counts) <= N * 0.80 and k > N * 0.1:
                save_phase_history[k][seed] = phase_history
                save_init_state_failed[k][seed] = init_clock_state
            else:
                save_init_state_success[k][seed] = init_clock_state
    
    save_flash_counts = pd.DataFrame(save_flash_counts)
    
    if update_noise is not None:
        noise_str = f"_update_noise={update_noise}"
    else:
        noise_str = ""
    
    save_flash_counts.to_csv(
        f"{path}/N={N}_clock_lnegth={clock_length}_T={T}_flash_proportion={flash_proportion}{noise_str}_k_regular_graph_flash_counts.csv",
        index=False)
    
    with open(
        f"{path}/N={N}_clock_lnegth={clock_length}_T={T}_flash_proportion={flash_proportion}{noise_str}_k_regular_graph_phase_history.pkl",
        'wb') as f:
        pickle.dump(save_phase_history, f)
    
    with open(
        f"{path}/N={N}_clock_lnegth={clock_length}_T={T}_flash_proportion={flash_proportion}{noise_str}_k_regular_graph_init_state_failed.pkl",
        'wb') as f:
        pickle.dump(save_init_state_failed, f)
    
    with open(
        f"{path}/N={N}_clock_lnegth={clock_length}_T={T}_flash_proportion={flash_proportion}{noise_str}_k_regular_graph_init_state_sucess.pkl",
        'wb') as f:
        pickle.dump(save_init_state_success, f)
