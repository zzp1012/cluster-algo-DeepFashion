# !/bin/bash

python main.py --save_root ../outs/predictions/kmeans \
               --seed 0 \
               --model kmeans \
               --n_clusters 3991 \
               -v

python main.py --save_root ../outs/predictions/HAC \
               --seed 0 \
               --model HAC \
               --n_clusters 3991 \
               -v