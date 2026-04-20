import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('QtAgg')

# Ns=(90 110 110 180 190)
# Cs=(26 14 30 26 14)
N = 180
C=54
T=10000
# path_to_experiment = f'/Volumes/Data/other/2026_firefly_synchronization/transition_experiment_2_local/N={N}_C={C}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.0_k_regular_graph_transition_flash_counts.pkl'
#
# data = pd.read_pickle(path_to_experiment)
#
# print(data.keys())
# for param in [0.05, 0.1, 0.2]:
#     for final_seed in data[int(N - N * param)].keys():
#     # for i in [0,10,20,30,40,50,60,70,80,90]:
#         print(data[int(N - N * param)][final_seed])
#         plt.plot(data[int(N - N * param)][final_seed])
#         plt.title(f"{final_seed}-{param}")
#         plt.show()
#     # print(data[int(N - N * param)])

N = 130
# path_to_experiment = '/Volumes/Data/other/2026_firefly_synchronization/r_com_range_2_local/flash_proportion=0.5_qr_threshold=0.5_update_noise=0.05/N=130_C=70_T=10000_r_com_range_flash_counts.pkl'
path_to_experiment = '/Volumes/Data/tmp_cluster/N=130_C=70_T=10000_r_com_range_flash_counts.pkl'
data = pd.read_pickle(path_to_experiment)

tmp_res = []
full_communication_data = pd.DataFrame(columns=['1.5'])
for r in [1.5]:
    for seed in range(1000):
        if np.max(data[1.5][seed]) < N:
            plt.plot(data[r][seed])
            # tmp_res.append(np.max(data[1.5][seed]))
            plt.show()
    # full_communication_data['1.5'] = tmp_res
    # print((full_communication_data.values < N).sum())
    # print((tmp_res < N))
    # save_flash_counts[N][C] = (full_communication_data.values < N).sum() / full_communication_data.shape[0]
