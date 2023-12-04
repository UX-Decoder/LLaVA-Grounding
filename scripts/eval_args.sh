root_name=$1
name=${3:-'answer'}
mkdir -p vqa/reviews/coco2014_val80/${root_name}/
python llava/eval/eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl \
    /comp_robot/zhanghao/model/${root_name}/checkpoint-${2}/${name}.json  \
    --rule llava/eval/table/rule.json \
    --output vqa/reviews/coco2014_val80/${root_name}/gpt4_text_${2}_${name}.jsonl