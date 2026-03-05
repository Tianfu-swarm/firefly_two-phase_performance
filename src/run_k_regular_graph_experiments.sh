n_seeds=1000
T=1000
k_range=(0 1 2 3 4 5 6 7 8 9 10 15 20 30 40 50 60 70 80 90 100)
Ns=(50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)  # number of fireflies
Cs=(10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70)
update_noises=(0.0 0.01 0.05 0.1 0.2 1)


for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for update_noise in "${update_noises[@]}"; do
      python k_regular_graph_experiments.py \
      --n_seeds $n_seeds \
      --N $N \
      --C $C \
      --update_noise $update_noise \
      --T $T \
      --k_range "${k_range[@]}"
    done
  done
done