#!/bin/bash

N_TRAIN=400
N_DEV=1000
N_TEST=1000
TRAIN_BZ=4

OUTPUT_PATH="."

# Set --checkpoint_dir to load pre-trained model from local
# Remove --new_design to test with original P-tuning
command='(python -u tuning.py \
  --model_name rinna/japanese-gpt-1b \
  --out_dir ${OUTPUT_PATH}/p_tuning_out \
  --dataset RCQA \
  --train_batch_size ${TRAIN_BZ} \
  --n_train_max ${N_TRAIN} \
  --n_dev_max ${N_DEV} \
  --n_test_max ${N_TEST} \
  --template \(3,3\) \
  --lr $LR \
  --epochs 60 \
  --early_stop 20 \
  --new_design \
  --new_num_mlp 1 \
  --new_residual \
  --new_random_init \
)'

if [ ! -d "logs" ]; then
  mkdir "logs"
fi

LRS=(6e-6 7e-6 8e-6 9e-6)
for LR in "${LRS[@]}"; do
  export LR N_TRAIN N_DEV N_TEST OUTPUT_PATH
  script -c "${command}" logs/grid_rcqa_new_temp3_3_lr"${LR}"_batch"${TRAIN_BZ}"_train"${N_TRAIN}"_dev"${N_DEV}"_test"${N_TEST}".txt
done
