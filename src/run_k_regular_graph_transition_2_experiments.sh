n_seeds=10
graph_seeds=10
T=5000
reduce_full_k_by=(0.0 0.05 0.1 0.2)  # % of links to be removed 0.05 = 5 %  --  0.1 0.2 0.3
Ns=(90 110 110 180 190)
Cs=(26 14 30 26 14)
update_noises=(0.0)

for i in "${!Ns[@]}"; do
  N=${Ns[$i]}
  C=${Cs[$i]}

  for update_noise in "${update_noises[@]}"; do
    python k_regular_graph_transition_2_experiments.py \
      --graph_seeds $graph_seeds \
      --n_seeds $n_seeds \
      --N $N \
      --C $C \
      --T $T \
      --t_switch 1000 \
      --update_noise $update_noise \
      --reduce_full_k_by "${reduce_full_k_by[@]}"
  done
done