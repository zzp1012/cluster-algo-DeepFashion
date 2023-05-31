# !/bin/bash

time=$(date +'%r')
echo "Times is" $time

python main.py --save_root ../outs/predictions/kmeans \
               --model kmeans \
               --n_clusters 3991 \
               -v

time=$(date +'%r')
echo "Times is" $time

python main.py --save_root ../outs/predictions/HAC \
               --model HAC \
               --n_clusters 3991 \
               -v

time=$(date +'%r')
echo "Times is" $time