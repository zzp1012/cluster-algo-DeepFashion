# !/bin/bash

python main.py --save_root ../outs/predictions/kmeans \
               --model kmeans \
               --n_clusters 3991 \
               -v

python main.py --save_root ../outs/predictions/HAC \
               --model HAC \
               --n_clusters 3991 \
               -v