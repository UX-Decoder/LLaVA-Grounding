#!/bin/bash
#SBATCH -J llll
#SBATCH -p cvr
#SBATCH --mem 1800G
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:hgx:8
#SBATCH -N 1
#SBATCH -e job-%j.err
#SBATCH -o job-%j.out

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########
out_dir=/comp_robot/zhanghao/model/llava_stage1_new/
mkdir -p $out_dir
echo $out_dir/log
deepspeed llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /comp_robot/zhanghao/ckpts/vicuna/vicuna-7b-v1.3/ \
    --version $PROMPT_VERSION \
    --data_path /comp_robot/lihongyang/code/VLLMs/LLaVA_new/data_preparation/data/flickr30k_train.json \
    --image_folder /comp_robot/cv_public_dataset/goldg/flickr30k_entities/train/ \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $out_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb >> $out_dir/log 2>&1
