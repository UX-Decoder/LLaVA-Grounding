python llava/eval/model_vqa.py \
    --model-path $1 \
    --question-file \
    /comp_robot/liushilong/data/llava-bench-in-the-wild/llava-bench-in-the-wild/questions.jsonl \
    --image-folder \
    /comp_robot/liushilong/data/llava-bench-in-the-wild/llava-bench-in-the-wild/images \
    --answers-file \
    $1/answer_wild.json