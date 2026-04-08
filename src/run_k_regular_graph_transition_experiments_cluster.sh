n_seeds=10
graph_seeds=100
T=10000
reduce_full_k_by=(0.05 0.1 0.2 0.3)  # % of links to be removed 0.05 = 5 %
Ns=(50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)  # number of fireflies
Cs=(10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70)  # 50 seems to be the upper limit considering T=1000
update_noises=(0.0)  #  0.1 0.2 1


for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for update_noise in "${update_noises[@]}"; do
      python k_regular_graph_transition_experiments.py \
      --save_dir "/abyss/home/results/fireflies" \
      --n_seeds $n_seeds \
      --graph_seeds $graph_seeds \
      --N $N \
      --C $C \
      --T $T \
      --t_switch 1000 \
      --update_noise $update_noise \
      --reduce_full_k_by "${reduce_full_k_by[@]}"
    done
  done
done