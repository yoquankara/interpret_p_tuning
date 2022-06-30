#!/bin/bash

if [ ! -d "logs" ]; then
  mkdir "logs"
fi

N_TRAIN=400
N_DEV=1000
N_TEST=1000

OUTPUT_PATH="."
P_TUNING_CKPT="<path to p_tuning checkpoint>"

command='(python -u tuning.py \
  --model_name rinna/japanese-gpt-1b \
  --out_dir ${OUTPUT_PATH}/p_tuning_out \
  --dataset RCQA \
  --n_train_max ${N_TRAIN} \
  --n_dev_max ${N_DEV} \
  --n_test_max ${N_TEST} \
  --template \(3,3\) \
  --lr $LR \
  --epochs 60 \
  --early_stop 20 \
  --only_evaluate \
  --print_topk \
  --load "${P_TUNING_CKPT}" \
  --eval_metrics_to_file logs/eval_rcqa_origin.tsv
)'

DUMMY_LRS=(6e-6)
for LR in "${DUMMY_LRS[@]}"; do
  export LR N_TRAIN N_DEV N_TEST OUTPUT_PATH P_TUNING_CKPT
  script -c "${command}"
done
