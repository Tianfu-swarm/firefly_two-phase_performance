import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def cast_to_python_types_and_normalize(d):
    new_dict = {}
    
    for key, value in d.items():
        # convert tuple key to pure Python ints
        new_key = tuple(int(x) for x in key)
        
        # convert nested dictionary
        new_value = {
            "counter": int(value["counter"]),
            "distribution": value["distribution"] / value["counter"]
        }
        
        new_dict[new_key] = new_value
    
    return new_dict


C = 50



with open(f'/Volumes/Data/other/2026_firefly_synchronization/N=100_clock_lnegth={C}_T=1000_flash_proportion=0.5_k_regular_graph_phase_history.pkl', 'rb') as f:
    data = pickle.load(f)
    
print(data.keys())
data = data[100]

print(f"total runs that didnt synchronize: {len(data)}")

known_patterns = {}

for run in data:
    final_distribution = run[-1, :]
    values, counts = np.unique(final_distribution, return_counts=True)
    distribution = counts / counts.sum()   # normalize to probabilities
    
    p = tuple(np.diff(values))
    
    if p in known_patterns:
        known_patterns[p]["counter"] += 1
        known_patterns[p]["distribution"] += distribution
    else:
        print(values)  # unique values
        print(distribution)  # probability of each value
        known_patterns[p] = {"counter": 1, "distribution": distribution}

# known_patterns = {
#     tuple(int(x) for x in key): int(value)
#     for key, value in known_patterns.items()
# }
known_patterns = cast_to_python_types_and_normalize(known_patterns)
print(known_patterns)

plt.figure()
for k in known_patterns.keys():
    print(k)
    x = np.concatenate(([0], np.cumsum(k)))
    y = known_patterns[k]["distribution"]
    plt.plot(x, y, marker='o', label=f"Pattern: {k} (n={known_patterns[k]['counter']})")
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel(r"Average distribution [N$_\text{sub}$ / N]")
plt.ylim(0, 1)
plt.xlim(0, C-1)
plt.show()

colorlist = list(plt.cm.hsv(np.linspace(0, 1, C, endpoint=False)))
for k in known_patterns.keys():
    plt.figure()
    starting = 0
    for i, j in enumerate(k):
        end = starting + C/2
        if starting >= 0 and end <= C:
            plt.barh(
                y=len(k)-i,
                width=C/2,
                left=starting,
                color=colorlist[i]
            )
        elif starting >= 0 and end > C:
            first_len = C - starting
            second_len = end - C
            
            # part 1: from i to CLOCK_LENGTH
            plt.barh(
                y=len(k)-i,
                width=first_len,
                left=starting,
                color=colorlist[i]
            )
            
            # part 2: from 0 onward
            plt.barh(
                y=len(k)-i,
                width=second_len,
                left=0,
                color=colorlist[i]
            )
        else:
            first_len = C + starting
            second_len = end
            
            # part 1: from i to CLOCK_LENGTH
            plt.barh(
                y=len(k)-i,
                width=first_len,
                left=C + starting,
                color=colorlist[i]
            )
            
            # part 2: from 0 onward
            plt.barh(
                y=len(k)-i,
                width=second_len,
                left=0,
                color=colorlist[i]
            )
        starting += j
    plt.xlim(0, C)
    plt.xlabel("Clock")
    plt.ylabel("Subgroups")
    plt.show()

    
    