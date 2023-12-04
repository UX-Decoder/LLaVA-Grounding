python llava/eval/eval_gpt_review_bench.py \
  --question /comp_robot/liushilong/data/llava-bench-in-the-wild/llava-bench-in-the-wild/questions.jsonl \
  --context /comp_robot/liushilong/data/llava-bench-in-the-wild/llava-bench-in-the-wild/context.jsonl \
  --answer-list \
  /comp_robot/liushilong/data/llava-bench-in-the-wild/llava-bench-in-the-wild/answers_gpt4.jsonl \
  $1 \
  --rule llava/eval/table/rule.json \
  --output vqa/reviews/llava_wild/gpt4_text_${2}.jsonl