model=/comp_robot/liushilong/data/LLAVA/LLAVA_7b
#model=/comp_robot/zhanghao/ckpts/llava_1_5/llava-v1.5-13b/
python llava/eval/model_vqa.py \
    --model-path $model \
    --question-file \
    /comp_robot/zhanghao/code/all/GPT4V_eval/gpt4v_data/llava/ref_question.jsonl \
    --image-folder \
    /comp_robot/zhanghao/code/all/GPT4V_eval/masks/new/refcoco_eval_pred_coloright_data100_new \
    --answers-file \
    /comp_robot/zhanghao/code/all/GPT4V_eval/result/answer.json
