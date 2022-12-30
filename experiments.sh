#!/bin/bash

while getopts g:s:d:t:r:o: flag; do
  case "${flag}" in
  g) GPUS=${OPTARG} ;;
  s) SEED=${OPTARG} ;;
  d) DATASET=${OPTARG} ;;
  t) NOISE_TYPE=${OPTARG} ;;
  r) NOISE_RATE=${OPTARG} ;;
  o) WORK_DIR=${OPTARG} ;;
  esac
done

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPUS

NOISE_RATE_PERS=$(python -c "print(int($NOISE_RATE*100))")
HAS_ANSWER=$(python -c "print('${DATASET}'=='newsqa')")

echo "================================================================================="
echo "GPUS:${GPUS}"
echo "SEED:${SEED}"
echo "DATASET:${DATASET}"
echo "NOISE_TYPE:${NOISE_TYPE}"
echo "NOISE_RATE=${NOISE_RATE}"
echo "NOISE_RATE_PERS=${NOISE_RATE_PERS}"
echo "HAS_ANSWER=${HAS_ANSWER}"
echo "================================================================================="

case $NOISE_TYPE in
"struct")
  p1="${DATASET}_noisy_TQpaired_Astructured_start"
  ;;
"unrel")
  p1="${DATASET}_noisy_corrAT_incorrQ"
  ;;
"rand")
  p1="${DATASET}_noisy_TQpair_Arand"
  ;;
*)
  echo "Incorrect noise type"
  ;;
esac

echo "================================================================================="
echo "START WRITING INPUTS IDS"
echo "================================================================================="

# write input ids
python main.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --output_dir ${WORK_DIR}/gr_weights_analysis_${NOISE_TYPE} \
  --cache_dir ${WORK_DIR}/gr_weights_analysis_${NOISE_TYPE} \
  --overwrite_output_dir \
  --overwrite_cache \
  --do_lower_case \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 24 \
  --learning_rate 2.5e-6 \
  --max_seq_length 384 \
  --num_train_epochs 5 \
  --doc_stride 128 \
  --threads 10 \
  --train_file ${WORK_DIR}/datasets_noise_${DATASET}/${p1}/train_${DATASET}_${NOISE_RATE}.json \
  --do_train \
  --logging_steps 100 \
  --gradients_dir ${WORK_DIR}/grad_w_b1_${SEED}_${DATASET}_${NOISE_TYPE}${NOISE_RATE_PERS} \
  --data_dir . \
  --output_grads_weights_ids true \
  --seed ${SEED} \
  --filter_negative ${HAS_ANSWER} \
  --token_ids_retreive

echo "================================================================================="
echo "START MAPPING INPUTS IDS"
echo "================================================================================="

python ./helpers/map_ids.py \
  --seed $SEED \
  --dataset $DATASET \
  --noise_type $NOISE_TYPE \
  --noise_rate $NOISE_RATE \
  --working_dir $WORK_DIR

echo "================================================================================="
echo "START WRITING GRADS & WEIGHTS"
echo "================================================================================="

python main.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --output_dir ${WORK_DIR}/gr_weights_analysis_${NOISE_TYPE} \
  --cache_dir ${WORK_DIR}/gr_weights_analysis_${NOISE_TYPE} \
  --overwrite_output_dir \
  --overwrite_cache \
  --do_lower_case \
  --per_gpu_train_batch_size 1 \
  --per_gpu_eval_batch_size 24 \
  --learning_rate 2.5e-6 \
  --max_seq_length 384 \
  --num_train_epochs 5 \
  --doc_stride 128 \
  --threads 10 \
  --train_file ${WORK_DIR}/datasets_noise_${DATASET}/${p1}/train_${DATASET}_${NOISE_RATE}.json \
  --do_train \
  --logging_steps 100 \
  --gradients_dir ${WORK_DIR}/grad_w_b1_${SEED}_${DATASET}_${NOISE_TYPE}${NOISE_RATE_PERS} \
  --data_dir . \
  --output_grads_weights_ids true \
  --seed $SEED \
  --p_pickle is_clean_${DATASET}_${NOISE_TYPE}${NOISE_RATE_PERS}_b1_seed${SEED}_5ep.pkl \
  --filter_negative $HAS_ANSWER

echo "================================================================================="
echo "FINISH"
echo "================================================================================="
