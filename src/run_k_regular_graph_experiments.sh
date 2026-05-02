T=10000
#k_range=()
Ns=(50 60 70 80 90)  # 110 120 130 140 150 160 170 180 190 200
Cs=(10 14 18 22 26 30 34)  #  54 58 62 66 70
update_noises=(0.0)


for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for update_noise in "${update_noises[@]}"; do
      python k_regular_graph_2_experiments.py \
      --graph_seeds 10 \
      --N $N \
      --C $C \
      --update_noise $update_noise \
      --T $T \
      --k_range "$N" \
#      --save_dir "/home/till/PycharmProjects/firefly_two-phase_performance/results"
#      --n_seeds $n_seeds \
    done
  done
done
