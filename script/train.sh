#!/bin/bash
export PYTHONPATH=.
code_length=256
nl_length=128
moco_k=3200  #1024
moco_m=0.999
lr=2e-5
moco_t=0.07

batch_size=40
max_steps=100000
save_steps=1000

base_model=microsoft/unixcoder-base
CUDA_VISIBLE_DEVICES=0,1,2,3
dataset=Train

function fine-tune-remix () {
    output_dir=./saved_models/fine_tune-unixcoder-${dataset}-${nl_length}-${code_length}/
    mkdir -p $output_dir
    echo ${output_dir}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run.py   \
        --dataset ${dataset} \
        --save_steps  ${save_steps} \
        --moco_m ${moco_m} --moco_t ${moco_t}  \
        --output_dir ${output_dir}  \
        --moco_k ${moco_k} \
        --config_name=${base_model}  \
        --model_name_or_path=${base_model} \
        --tokenizer_name=${base_model} \
        --do_train\
        --do_test\
        --train_data_file=../dataset/${dataset}/train.jsonl\
        --eval_data_file=../dataset/${dataset}/val.jsonl \
        --code_length ${code_length} \
        --nl_length ${nl_length} \
        --train_batch_size ${batch_size} \
        --eval_batch_size 128 \
        --learning_rate ${lr} \
        --patience 20 \
        --seed 123456 2>&1| tee ${output_dir}/running.log
}

fine-tune-remix
