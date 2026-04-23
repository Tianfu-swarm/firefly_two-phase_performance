import pandas as pd
import numpy as np

T = 10000
Ns = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
Cs = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]  # , 54, 58, 62, 66, 70
base_dir = '/home/till/PycharmProjects/firefly_two-phase_performance/results'
# load data
for noise_level in [0.8, 0.9]:  # ,0.05, 0.1, 0.2
    save_flash_counts = {}
    for N in Ns:
        save_flash_counts[N] = {}
        for C in Cs:
            save_flash_counts[N][C] = 0.0
            try:
                path = f'{base_dir}/r_com_range_2_local/flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}/N={N}_C={C}_T=10000_r_com_range_flash_counts.pkl'
                data = pd.read_pickle(path)
                for run in data[1.5].keys():
                    if np.max(data[1.5][run]) <= N * 0.85:
                        save_flash_counts[N][C] +=1
            except FileNotFoundError:
                save_flash_counts[N][C] = np.nan
                print(f'File {N}/{C} not found.')
            
    heatmap = np.zeros((len(Cs), len(Ns)))
    for i, C in enumerate(Cs):
        for j, N in enumerate(Ns):
            heatmap[i, j] = save_flash_counts.get(N, {}).get(C, np.nan)
    heatmap_path = f'{base_dir}/heatmap_noise_lower_phase_{noise_level}.npz'
    np.savez_compressed(heatmap_path, arr=heatmap)