# !/bin/bash

python main.py --save_root ../outs/predictions/kmeans \
               --seed 0 \
               --model kmeans \
               --n_clusters 3991 \
               -v

python main.py --save_root ../outs/predictions/spectral \
               --seed 0 \
               --model spectral \
               --n_clusters 2504 \
               -v