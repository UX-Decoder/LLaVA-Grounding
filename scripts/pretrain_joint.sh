# Uncomment and set the following variables correspondingly to run this script:

# MODEL_VERSION=vicuna-v1-3-7b
# MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=v1
########### DO NOT CHANGE ###########
out_dir=output/llava_grounding_stage1
mkdir -p $out_dir
echo $out_dir/log
export DATASET=datasets/

n_gpu=4

deepspeed --include=localhost:1,2,3,7 llava/train/train_joint_1st.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ckpts/vicuna/vicuna-7b-v1.3/ \
    --version $PROMPT_VERSION \
    --data_path datasets/llava/annotations/cap600k_brackets_all.json \
    --image_folder datasets/ConceptualCaptionsFiltered/ \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter output/llava_stage1/mm_projector.bin \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $out_dir \
    --max_steps 30000 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 100 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --config_file \
    configs/openseed/openseed_swint_lang_joint.yaml \
    --opt \
    flickr.TRAIN.BATCH_SIZE_TOTAL=8,COCO.TRAIN.BATCH_SIZE_TOTAL=24,MODEL.WEIGHTS=ckpts/openseed_o365.pt \
    >> $out_dir/log 2>&1
