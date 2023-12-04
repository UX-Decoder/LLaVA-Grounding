#!/bin/bash
#SBATCH --job-name=pretrain # create a short name for your job
#SBATCH -p batch
##SBATCH -x dgx[047,037,021] # 排除一些节点
#SBATCH --reservation=ccp
#SBATCH --qos ccp
#SBATCH -N 1 # node count
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task 100 # cpu-cores per task (>1 if multi-threaded tasks),--cpus-per-task
#SBATCH --gpus-per-node 8 # total cpus for job

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
    --data_path /comp_robot/cv_public_dataset/CC12M_zh/LLaVA-CC3M-Pretrain-595K/chat_new.json \
    --image_folder /comp_robot/cv_public_dataset/ConceptualCaptionsFiltered/ \
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