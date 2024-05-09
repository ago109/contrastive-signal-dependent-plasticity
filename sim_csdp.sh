#!/bin/bash

################################################################################
# Simulate CSDP on target pattern database
################################################################################
GPU_ID=0
SEEDS=(1234)
DATASET="mnist"
DATA_DIR="data/"$DATASET

ALGO_TYPE="supervised"
NUM_ITER=1 #30
NZ1=1024
NZ2=1024

EXP_DIR="exp_"$ALGO_TYPE"_"$DATASET ## make exp dir
## run simulation/experiment
rm -r "$EXP_DIR/"* ## clear out experimental directory
for SEED in "${SEEDS[@]}"
do
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_csdp.py  --dataX="$DATA_DIR/trainX.npy" \
                                                     --dataY="$DATA_DIR/trainY.npy" \
                                                     --devX="$DATA_DIR/validX.npy" \
                                                     --devY="$DATA_DIR/validY.npy" \
                                                     --algo_type=$ALGO_TYPE \
                                                     --num_iter=$NUM_ITER \
                                                     --verbosity=0 --seed=$SEED \
                                                     --exp_dir=$EXP_DIR --nZ1=$NZ1 --nZ2=$NZ2
done
