#!/bin/bash
#SBATCH -J llll
#SBATCH -p cvr
#SBATCH --mem 800G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:hgx:4
#SBATCH -N 1
#SBATCH -e job-%j.err
#SBATCH -o job-%j.out

# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=v1
########### DO NOT CHANGE ###########
out_dir=/comp_robot/zhanghao/model/llava_stage1_new_gd_flickr_continue_from_vg_4gpu/
mkdir -p $out_dir
echo $out_dir/log
export DATASET=/comp_robot/cv_public_dataset/ DETECTRON2_DATASETS=/comp_robot/cv_public_dataset/ META_ROOT=/comp_robot/zhanghao/datasets/imagenet22k/  PYTHONPATH=/comp_robot/zhanghao/code/all/LLaVA_new:$PYTHONPATH PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/comp_robot/zhanghao/cuda-11.7/lib64/:$LD_LIBRARY_PATH
deepspeed --include=localhost:2,1,4,5 --master_port 29503 llava/train/train_hao_seg_flickr.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path /comp_robot/zhanghao/ckpts/vicuna/vicuna-7b-v1.3/ \
    --whole_model /comp_robot/zhanghao/model/llava_stage1_new_gd_refcoco_fixed_vg/ \
    --load_vision True \
    --version $PROMPT_VERSION \
    --load_optimizer_states False \
    --load_lr_scheduler_states False \
    --data_path /comp_robot/zhanghao/datasets/llava/cap600k_brackets_all.json \
    --image_folder /comp_robot/cv_public_dataset/ConceptualCaptionsFiltered/ \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /comp_robot/zhanghao/model/llava_stage1_new/mm_projector.bin \
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
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --max_steps 10000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --config_file \
    configs/openseed/openseed_swint_lang_flickr.yaml \
    --opt \
    flickr.TEST.BATCH_SIZE_TOTAL=32,flickr.TRAIN.BATCH_SIZE_TOTAL=32,MODEL.WEIGHTS=/comp_robot/zhanghao/ckpts/openseed_o365.pt \
    >> $out_dir/log 2>&1
