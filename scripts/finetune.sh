# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################
out_dir=output/llava_grounding_stage2
load=output/llava_grounding_stage1
mkdir -p $out_dir
echo $out_dir/log
export DATASET=datasets/

num_gpu=8
bs=$(( 8 * $num_gpu ))
deepspeed llava/train/train_joint_2st.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ckpts/vicuna/vicuna-7b-v1.3/ \
    --whole_model $load \
    --load_model True \
    --version $PROMPT_VERSION \
    --data_path datasets/llava/annotations/llava_instruct_150k.json \
    --image_folder datasets/coco/train2017/ \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter output/llava_stage1/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $out_dir \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2400 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --max_steps 10000 \
    --config_file \
    configs/openseed/openseed_swint_lang_joint_2st.yaml \
    --opt \
    MODEL.DECODER.WEIGHT_MULTIPLIER=0.1,MODEL.DECODER.COST_CLASS_WEIGHT=4.0,flickr.TRAIN.BATCH_SIZE_TOTAL=6,coco_instruct.TEST.BATCH_SIZE_TOTAL=${bs},coco_instruct.TRAIN.BATCH_SIZE_TOTAL=${bs},MODEL.WEIGHTS=ckpts/openseed_o365.pt \
    >> $out_dir/log 2>&1
