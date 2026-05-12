n_seeds=1
graph_seeds=1000
T=1000
k_range=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200)
#k_range=(100)
Ns=(100)  # number of fireflies  50 60 70 80 90 100 110 120 130 140 150 160 170 180 190
Cs=(34)  # 50 seems to be the upper limit considering T=1000  10 14 18 22 26 30 34 38 42 46 54 58 62 66 70
update_noises=(0.0)  # 0.1 0.2 1.0
flash_proportions=(0.1 0.2 0.33 0.4 0.5 0.6)  # 0.1 0.2 0.33 0.4 0.5 0.6
qr_thresholds=(0.1 0.2 0.33 0.4 0.5 0.6)  # 0.1 0.2 0.33 0.4 0.5 0.6

for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for update_noise in "${update_noises[@]}"; do
      for flash_proportion in "${flash_proportions[@]}"; do
        for qr_threshold in "${qr_thresholds[@]}"; do
          python k_regular_graph_experiments.py \
          --n_seeds $n_seeds \
          --graph_seeds $graph_seeds \
          --save_dir "/Volumes/Data/other/2026_firefly_synchronization/qr_f_experiments_k_graph" \
          --N $N \
          --C $C \
          --update_noise $update_noise \
          --flash_proportion $flash_proportion \
          --qr_threshold $qr_threshold \
          --T $T \
          --k_range "${k_range[@]}"
        done
      done
    done
  done
done