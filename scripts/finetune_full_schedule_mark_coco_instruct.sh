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

################## VICUNA ##################
PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################
out_dir=/comp_robot/zhanghao/model/llava_stage2_new_flickr_mark_coco_instruct/
mkdir -p $out_dir
echo $out_dir/log
im_path=/comp_robot/zhanghao/code/all/GPT4V_eval/gpt4v_data/coco_bs5_alpha_0.1_instance_from_1_font_size_20_instance_only
json_path=/comp_robot/lihongyang/code/VLLMs/LLaVA_new/merged_v3_fixed_inter_marker.json
export LD_LIBRARY_PATH=/comp_robot/zhanghao/cuda-11.7/lib64/:$LD_LIBRARY_PATH
deepspeed llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /comp_robot/zhanghao/ckpts/vicuna/vicuna-7b-v1.3/ \
    --version $PROMPT_VERSION \
    --data_path $json_path \
    --image_folder $im_path \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /comp_robot/zhanghao/model/llava_stage1_new/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $out_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
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
#--tune_mm_mlp_adapter True
#  --include=localhost:2,3,4,5 --master_port 29503