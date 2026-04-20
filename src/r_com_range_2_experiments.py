import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
import argparse

from simulation import simulate_fireflies_r_communication_range
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--save_dir', type=str, default="/Volumes/Data/other/2026_firefly_synchronization")
    arg_parser.add_argument('--n_seeds', type=int, default=1000)
    arg_parser.add_argument('--graph_seeds', type=int, default=1)
    # simulation params
    arg_parser.add_argument('--N', type=int, default=100)  # number of fireflies
    arg_parser.add_argument('--C', type=int, default=10)  # clock length (number of discrete phases)
    arg_parser.add_argument('--T', type=int, default=1000)  # number of time steps to simulate
    arg_parser.add_argument('--t_switch', type=int, default=1000)
    arg_parser.add_argument('--flash_proportion', type=float, default=0.5)  # how long to flash
    arg_parser.add_argument('--qr_threshold', type=float, default=0.5)  # how long to flash
    arg_parser.add_argument('--update_noise', type=float, default=0.0)  # how long to flash
    arg_parser.add_argument('--r_range', nargs="+", type=float,
                            default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    
    args = arg_parser.parse_args()
    
    # make a quick check if the results already exist and skip if they do
    flash_counts_path = (f"{args.save_dir}/r_com_range_2_local/"
                         f"flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/"
                         f"N={args.N}_C={args.C}_T={args.T}_r_com_range_flash_counts.pkl")
    if os.path.isfile(flash_counts_path):
        print(f"{flash_counts_path} already exists. skipping...")
        exit(0)
    
    run_params = []
    save_flash_counts = {}
    # save_phase_history = {}
    # save_init_state_failed = {}
    # save_init_state_success = {}
    # avg_num_neighbors = {}
    for r in args.r_range:
        save_flash_counts[r] = {}
        # avg_num_neighbors[r] = []
        # save_phase_history[r] = {}
        # save_init_state_failed[r] = np.zeros((args.n_seeds, args.N))
        # save_init_state_success[r] = np.zeros((args.n_seeds, args.N))
        for seed_graph in range(1):
            rng = np.random.default_rng(seed_graph)
            pos = rng.random((args.N, 2))
            dists = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
            communication_graph = ((dists < r) & (dists > 0)).astype(int)
            
            # avg_num_neighbors[r].append(float(np.mean(np.sum(communication_graph, axis=1) / (args.N - 1))))
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed)
                # phases = rng.integers(0, args.C, size=args.N)
                if args.N in [50, 60, 70, 80, 100, 120, 130, 140, 150, 160, 170, 200]:
                    phases = np.floor(np.linspace(0, args.C, args.N, endpoint=False)).astype(int)
                else:
                    phases = np.ceil(np.linspace(0, args.C, args.N, endpoint=False)).astype(int)
                
                run_params.append(
                    (args.N, args.C, phases, communication_graph, args.T, args.flash_proportion, args.qr_threshold, r,
                     seed))
    
    # avg_num_neighbors_df = pd.DataFrame(avg_num_neighbors)
    # avg_num_neighbors_df.to_csv(f"{args.save_dir}/flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/N={args.N}_C={args.C}_T={args.T}_r_com_range_avg_neighbors.csv",
    #     index=False)
    print(f"done setting up the parameters ...")
    print(
        f"Running: N={args.N} | C={args.C} | T={args.T} | flash_proportion={args.flash_proportion} | qr_threshold={args.qr_threshold} | update_noise={args.update_noise} | r_range={args.r_range}")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        futures = [
            executor.submit(simulate_fireflies_r_communication_range, N, clock_length, phases, communication_graph,
                            args.T, args.t_switch,
                            flash_proportion, qr_threshold, r, seed, args.update_noise)
            for (N, clock_length, phases, communication_graph, args.T, flash_proportion, qr_threshold, r, seed) in
            run_params
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            flash_counts, phase_history, groups_history, r, init_clock_state, seed = future.result()
            save_flash_counts[r][seed] = flash_counts
            # save_phase_history[r][seed] = phase_history
            # if np.max(flash_counts) <= args.N * 0.90 and r > args.N * 0.1:
            #     save_init_state_failed[r][seed] = init_clock_state
            # else:
            #     save_init_state_success[r][seed] = init_clock_state
    
    os.makedirs(f"{args.save_dir}/r_com_range_2_local/"
                f"flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/",
                exist_ok=True)
    
    with open(
        f"{args.save_dir}/r_com_range_2_local/"
        f"flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/"
        f"N={args.N}_C={args.C}_T={args.T}_r_com_range_flash_counts.pkl",
        'wb') as f:
        pickle.dump(save_flash_counts, f)
    
    # with open(
    #     f"{args.save_dir}/"
    #     f"flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/"
    #     f"N={args.N}_C={args.C}_T={args.T}_r_com_range_phase_history.pkl",
    #     'wb') as f:
    #     pickle.dump(save_phase_history, f)
    
    # with open(
    #     f"{args.save_dir}/N={args.N}_C={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_update_noise={args.update_noise}_r_com_range_init_state_failed.pkl",
    #     'wb') as f:
    #     pickle.dump(save_init_state_failed, f)
    #
    # with open(
    #     f"{args.save_dir}/N={args.N}_C={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_update_noise={args.update_noise}_r_com_range_init_state_sucess.pkl",
    #     'wb') as f:
    #     pickle.dump(save_init_state_success, f)
