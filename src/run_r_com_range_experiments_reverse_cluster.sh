n_seeds=1000
T=10000
r_range=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5) #
Ns=(200 190 180 170 160 150 140 130 120 110 100 90 80 70 60 50)  # number of fireflies
Cs=(70 66 62 58 54 50 46 42 38 34 30 26 22 18 14 10)  # 50 seems to be the upper limit considering T=1000
update_noises=(0.0 0.01 0.05 0.1 0.2)


for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for update_noise in "${update_noises[@]}"; do
      python r_com_range_experiments.py \
      --save_dir "/abyss/home/results/fireflies" \
      --n_seeds $n_seeds \
      --N $N \
      --C $C \
      --update_noise $update_noise \
      --T $T \
      --r_range "${r_range[@]}"
    done
  done
done