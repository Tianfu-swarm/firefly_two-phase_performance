n_seeds=1000
T=10000
r_range=(1.5)  #
Ns=(50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)  # number of fireflies
Cs=(10 14 18 22 26 30 34 38 42 46 50 54 58 62 66 70)  # 50 seems to be the upper limit considering T=1000
update_noises=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)  # 0.0 0.05 0.1 0.2


for N in "${Ns[@]}"; do
  for C in "${Cs[@]}"; do
    for update_noise in "${update_noises[@]}"; do
      python r_com_range_2_experiments.py \
      --save_dir "/home/till/PycharmProjects/firefly_two-phase_performance/results" \
      --n_seeds $n_seeds \
      --N $N \
      --C $C \
      --update_noise $update_noise \
      --T $T \
      --t_switch 1000 \
      --r_range "${r_range[@]}"
    done
  done
done