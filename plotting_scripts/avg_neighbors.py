import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

N = 100
k = pd.read_csv('/Volumes/Data/other/2026_firefly_synchronization/N=100_clock_lnegth=10_T=1000_flash_proportion=0.5_k_regular_graph_avg_neighbors.csv')
r = pd.read_csv('/Volumes/Data/other/2026_firefly_synchronization/N=100_clock_lnegth=10_T=1000_flash_proportion=0.5_r_com_range_avg_neighbors.csv')


fig, ax1 = plt.subplots(figsize=(8,6))

# Compute row means
k_row_means = k.mean(axis=0)
r_row_means = r.mean(axis=0)

# Plot k dataframe
ax1.plot(k_row_means, k.columns.astype(int) / N, 'o-', color='blue', label='k')
ax1.set_xlabel('Average number of neighbors [#neighbors / N]')
ax1.set_ylabel('k [k / N]', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for r
ax2 = ax1.twinx()
ax2.plot(r_row_means, r.columns, 's--', color='red', label='r')
ax2.set_ylabel('Communication range r', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# plt.title('Row means vs. column labels')
plt.show()