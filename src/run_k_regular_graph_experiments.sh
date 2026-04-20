n_seeds=10
T=10000
k_range=(0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200)
Ns=(100)  # number of fireflies
Cs=(10)  # 50 seems to be the upper limit considering T=1000 54 58 62 66 70
update_noises=(0.0)


for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for update_noise in "${update_noises[@]}"; do
      python k_regular_graph_2_experiments.py \
      --save_dir "/home/till/PycharmProjects/firefly_two-phase_performance/results" \
      --n_seeds $n_seeds \
      --N $N \
      --C $C \
      --update_noise $update_noise \
      --T $T \
      --k_range "${k_range[@]}"
    done
  done
done