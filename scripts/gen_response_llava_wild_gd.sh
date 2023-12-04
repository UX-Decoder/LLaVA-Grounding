python llava/eval/model_vqa.py \
    --model-path $1 \
    --question-file \
    playground/data/llava-bench-in-the-wild/questions_with_grounding.jsonl \
    --image-folder \
    /comp_robot/liushilong/data/llava-bench-in-the-wild/llava-bench-in-the-wild/images \
    --answers-file \
    $1/answer_wild_gd_${2}.json