python llava/eval/model_vqa.py \
    --model-path $1 \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /comp_robot/rentianhe/dataset/coco/val2014/ \
    --answers-file \
    $1/answer.json
