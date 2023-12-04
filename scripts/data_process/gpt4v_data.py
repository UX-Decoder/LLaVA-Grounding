import json
fake_ans="playground/data/coco2014_val_qa_eval/qa90_gpt4_v_answer.jsonl"
data_ls=[]
with open(fake_ans,'r') as f:
    for line in f:
        data=json.loads(line)
        data_ls.append(data)
# with open("/comp_robot/lihongyang/code/VLLMs/LLaVA_new/eval_gpt4/questions_answer.json") as f:
#     ...:     j=json.load(f)
gpt4v_res="/comp_robot/lihongyang/code/VLLMs/LLaVA_new/eval_gpt4/questions_answer.json"
with open(gpt4v_res,'r') as f:
    gpt4v=json.load(f)

for data in data_ls:
    im_id=data['question_id']//3
    type=data['category']
    gpt4v_ans=gpt4v[str(im_id)][type]['answer']
    data['text']=gpt4v_ans

with open("playground/data/coco2014_val_qa_eval/qa90_gpt4_v_answer.jsonl",'w') as f:
    for data in data_ls:
        json.dump(data,f)
        f.write('\n')