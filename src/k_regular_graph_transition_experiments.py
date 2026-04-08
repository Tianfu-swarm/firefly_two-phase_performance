import numpy as np
import os
import pandas as pd
import igraph as ig
import random
import pickle
from tqdm import tqdm
import argparse

from simulation import simulate_fireflies_k_regular_graph_transition
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--save_dir', type=str, default="/Volumes/Data/other/2026_firefly_synchronization")
    arg_parser.add_argument('--n_seeds', type=int, default=1)
    arg_parser.add_argument('--graph_seeds', type=int, default=2)
    # simulation params
    arg_parser.add_argument('--N', type=int, default=100)  # number of fireflies
    arg_parser.add_argument('--C', type=int, default=10)  # clock length (number of discrete phases)
    arg_parser.add_argument('--T', type=int, default=1000)  # number of time steps to simulate
    arg_parser.add_argument('--t_switch', type=int, default=100)  # number of time steps to simulate
    arg_parser.add_argument('--flash_proportion', type=float, default=0.5)  # how long to flash
    arg_parser.add_argument('--qr_threshold', type=float, default=0.5)  # how long to flash
    arg_parser.add_argument('--update_noise', type=float, default=0.0)  # how long to flash
    arg_parser.add_argument('--reduce_full_k_by', nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3])
    
    args = arg_parser.parse_args()
    
    # make a quick check if the results already exist and skip if they do
    flash_counts_path = f"{args.save_dir}/transition_experiment/N={args.N}_C={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_update_noise={args.update_noise}_k_regular_graph_flash_counts.pkl"
    if os.path.isfile(flash_counts_path):
        print(f"{flash_counts_path} already exists. skipping...")
        exit(0)
    
    # ensure k values are valid (k must be less than N for k-regular graph)
    args.k_reduced_range = [int(args.N - args.N * k_reduced) for k_reduced in args.reduce_full_k_by]
    # full communication to start
    communication_graph_1 = np.ones((args.N, args.N))
    
    run_params = []
    save_flash_counts = {}
    save_phase_history = {}
    save_init_state_failed = {}
    save_init_state_success = {}
    for k in args.k_reduced_range:
        save_flash_counts[k] = {}
        save_phase_history[k] = np.zeros((args.graph_seeds * args.n_seeds, args.N))
        save_init_state_failed[k] = np.zeros((args.graph_seeds * args.n_seeds, args.N))
        save_init_state_success[k] = np.zeros((args.graph_seeds * args.n_seeds, args.N))
        for seed_graph in range(args.graph_seeds):
            data = np.load(f"{args.save_dir}/pre_computed_graphs/N={args.N}_k={k}_seed={int(seed_graph)}.npz")
            communication_graph_2 = data["communication_graph"]
            # random.seed(seed_graph)
            # np.random.seed(seed_graph)
            # ig.set_random_number_generator(random)
            # try:
            #     G = ig.Graph.K_Regular(args.N, k)
            # except:
            #     print(f"Cannot generate k-regular graph with k={k} and N={args.N} (should be k > N). Skipping this k.")
            #     continue
            # print(f"done generating the graph for k={k} and seed={seed_graph}")
            # communication_graph_2 = np.array(G.get_adjacency().data)
            
            for seed in range(args.n_seeds):
                n_subgroups = args.C
                shifts = (args.C // n_subgroups)
                group_size_fill = args.N - (n_subgroups - 1) * (args.N // n_subgroups)
                # phases = np.concatenate([np.ones(args.N // n_subgroups, dtype=int) * i * shifts for i in range(n_subgroups-1)])
                # phases = np.concatenate([phases, np.ones(group_size_fill, dtype=int) * (n_subgroups-1) * (shifts)])
                # phases = np.concatenate([np.ones(args.N // n_subgroups, dtype=int) * shifts * 0,
                #                          np.ones(args.N // n_subgroups, dtype=int) * shifts * 1,
                #                          np.ones(args.N // n_subgroups, dtype=int) * shifts * 2,
                #                          np.ones(args.N // n_subgroups, dtype=int) * shifts * 3,])
                # phases = np.concatenate([phases, np.ones(group_size_fill, dtype=int) * shifts * 4])
                phases = np.floor(np.linspace(0, args.C, args.N, endpoint=False)).astype(int)
                # print(f"phases (shape: {phases.shape}): {phases}")
                # print(f"phase shift: {shifts}")
                final_seed = int(args.n_seeds * seed_graph + seed)
                run_params.append(
                    (args.N, args.C, phases, communication_graph_1, communication_graph_2, args.t_switch, args.T,
                     args.flash_proportion, k, final_seed))
    
    print(f"done setting up the parameters ...")
    print(
        f"Running: N={args.N} | C={args.C} | T={args.T} | flash_proportion={args.flash_proportion} | update_noise={args.update_noise} | k_reduced_range={args.k_reduced_range}")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        futures = [
            executor.submit(simulate_fireflies_k_regular_graph_transition, N, clock_length, phases,
                            communication_graph_1, communication_graph_2, t_switch, T,
                            flash_proportion, args.qr_threshold, k, seed, args.update_noise)
            for
            (N, clock_length, phases, communication_graph_1, communication_graph_2, t_switch, T, flash_proportion, k,
             seed) in run_params
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            flash_counts, phase_history, groups_history, k, init_clock_state, seed = future.result()
            save_flash_counts[k][seed] = flash_counts
            if np.max(flash_counts) <= args.N * 0.90 and k > args.N * 0.1:
                save_phase_history[k][seed] = phase_history
                save_init_state_failed[k][seed] = init_clock_state
            else:
                save_init_state_success[k][seed] = init_clock_state
    
    os.makedirs(f"{args.save_dir}/transition_experiment/", exist_ok=True)
    
    with open(
        f"{args.save_dir}/transition_experiment/N={args.N}_C={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}_k_regular_graph_transition_flash_counts.pkl",
        'wb') as f:
        pickle.dump(save_flash_counts, f)
    
    # with open(
    #     f"{args.save_dir}/transition_experiment/N={args.N}_C={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}_k_regular_graph_transition_phase_history.pkl",
    #     'wb') as f:
    #     pickle.dump(save_phase_history, f)

    # with open(
    #     f"{args.save_dir}/transition_experiment/N={args.N}_C={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}_k_regular_graph_transition_init_state_failed.pkl",
    #     'wb') as f:
    #     pickle.dump(save_init_state_failed, f)
    #
    # with open(
    #     f"{args.save_dir}/transition_experiment/N={args.N}_C={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_update_noise={args.update_noise}_k_regular_graph_transition_init_state_sucess.pkl",
    #     'wb') as f:
    #     pickle.dump(save_init_state_success, f)
