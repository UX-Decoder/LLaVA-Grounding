import json
question_file='playground/data/coco2014_val_qa_eval/qa90_questions.jsonl'
out_question_file='playground/data/coco2014_val_qa_eval/qa90_questions_with_grounding.jsonl'
with open(question_file) as f:
    questions = [json.loads(line) for line in f]
    for q in questions:
        q['text'] = q['text'] + ' (with grounding)'
    # questions = [q+" (with grounding)" for q in questions]
with open(out_question_file, 'w') as f:
    for q in questions:
        f.write(json.dumps(q) + '\n')