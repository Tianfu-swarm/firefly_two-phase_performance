n_seeds=10
T=10000
#k_range=()
Ns=(50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)  # number of fireflies
Cs=(10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70)  # 50 seems to be the upper limit considering T=1000
update_noises=(0.0)


for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for update_noise in "${update_noises[@]}"; do
      python k_regular_graph_2_experiments.py \
      --n_seeds $n_seeds \
      --graph_seeds 100 \
      --N $N \
      --C $C \
      --update_noise $update_noise \
      --T $T \
      --k_range "$N" \
      --save_dir "/home/till/PycharmProjects/firefly_two-phase_performance/results"
    done
  done
done