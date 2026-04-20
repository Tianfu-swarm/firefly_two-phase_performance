n_seeds=1000
T=10000
r_range=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)  #
Ns=(130)  # number of fireflies  50 60 70 80 90 100 110 120 130 140 150 160 170 180 190
Cs=(70)  # 50 seems to be the upper limit considering T=1000  10 14 18 22 26 30 34 38 42 46 54 58 62 66 70
update_noises=(0.05)  # 0.0 0.05 0.1 0.2


for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for update_noise in "${update_noises[@]}"; do
      python r_com_range_2_experiments.py \
      --n_seeds $n_seeds \
      --N $N \
      --C $C \
      --update_noise $update_noise \
      --T $T \
      --r_range "${r_range[@]}"
    done
  done
done