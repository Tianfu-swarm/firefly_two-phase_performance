import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

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
    
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--save_dir', type=str, default="/Volumes/Data/other/2026_firefly_synchronization")
    arg_parser.add_argument('--n_seeds', type=int, default=1000)
    arg_parser.add_argument('--graph_seeds', type=int, default=1)
    # simulation params
    arg_parser.add_argument('--N', type=int, default=100)  # number of fireflies
    arg_parser.add_argument('--C', type=int, default=10)  # clock length (number of discrete phases)
    arg_parser.add_argument('--T', type=int, default=1000)  # number of time steps to simulate
    arg_parser.add_argument('--flash_proportion', type=float, default=0.5)  # how long to flash
    arg_parser.add_argument('--update_noise', type=float, default=0.0)  # how long to flash
    arg_parser.add_argument('--r_range', nargs="+", type=int, default=[0, 10, 20, 30, 100])
    
    args = arg_parser.parse_args()
    
    run_params = []
    save_flash_counts = {}
    avg_num_neighbors = {}
    for r in args.r_range:
        save_flash_counts[r] = []
        avg_num_neighbors[r] = []
        for seed_graph in range(1):
            rng = np.random.default_rng(seed_graph)
            pos = rng.random((args.N, 2))
            dists = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
            communication_graph = ((dists < r) & (dists > 0)).astype(int)
            
            avg_num_neighbors[r].append(float(np.mean(np.sum(communication_graph, axis=1) / (args.N-1))))
            for seed in range(10000):
                rng = np.random.default_rng(seed)
                phases = rng.integers(0, args.clock_length, size=args.N)
                
                run_params.append((args.N, args.clock_length, phases, communication_graph, args.T, args.flash_proportion, r))
    
    avg_num_neighbors_df = pd.DataFrame(avg_num_neighbors)
    avg_num_neighbors_df.to_csv(f"{args.save_dir}/N={args.N}_clock_lnegth={args.clock_length}_T={args.T}_flash_proportion={args.flash_proportion}_r_com_range_avg_neighbors.csv",
        index=False)
    print(f"done setting up the parameters ...")
    print(
        f"Running: N={args.N} | C={args.C} | T={args.T} | flash_proportion={args.flash_proportion} | update_noise={args.update_noise} | r_range={args.k_range}")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        futures = [
            executor.submit(simulate_fireflies_communication_range, N, clock_length, phases, communication_graph, T,
                            flash_proportion, r, args.update_noise)
            for (N, clock_length, phases, communication_graph, T, flash_proportion, r) in run_params
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            flash_counts, phase_history, groups_history, r = future.result()
            save_flash_counts[r].append(np.max(flash_counts))
        
    save_flash_counts = pd.DataFrame(save_flash_counts)
    
    noise_str = f"_update_noise={args.update_noise}"
    
    save_flash_counts.to_csv(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.clock_length}_T={args.T}_flash_proportion={args.flash_proportion}{noise_str}_r_com_range_flash_counts.csv",
        index=False)
    
    
