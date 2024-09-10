

DATASET=$1 # Available datasets: qata_cov19_v2_2, monuseg_2,

python train.py --exp experiments/${DATASET}/ddpm.json

