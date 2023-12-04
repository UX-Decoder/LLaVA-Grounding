import json
ori_question="playground/data/coco2014_val_qa_eval/qa90_questions.jsonl"
ori_answer="playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl"
new_conv="/comp_robot/cv_public_dataset/coco/annotations/llava_bench_qa90_gpt4_conv_end.json"
#id, image, conversations, gd_ls
with open(ori_question) as f:
    questions = [json.loads(line) for line in f]
with open(ori_answer) as f:
    answers = [json.loads(line) for line in f]
with open(new_conv) as f:
    convs=json.load(f)

for q,a in zip(questions,answers):
    succ=False
    for c in convs:
        if c['conversations'][0]['value'].startswith(q['text']):
            c['question_id']=q['question_id']
            succ=True
            break
    if not succ:
        convs.append({'question_id':q['question_id'],'image':q['image'],'id':q['image'][:-4],'conversations':[{"from": "human",'value':q['text']},{"from": "gpt",'value':a['text']}]})
assert len(convs)==90
out="/comp_robot/cv_public_dataset/coco/annotations/llava_bench_qa90_gpt4_conv_end_hy.json"
with open(out,'w') as f:
    json.dump(convs,f)
