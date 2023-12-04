import json
new_answer_file="/comp_robot/zhanghao/code/all/LLaVA_0505/playground/data/coco2014_val_qa_eval/qa90_gpt4_answer_v2.jsonl"
ls=[]
for line in open(new_answer_file,'r').readlines():
    a=json.loads(line)
    ls.append(a['text'])
    ls.append(a['new_answer'])
    ls.append("##############################################")
    # a['new_answer']
out="tmp.txt"
with open(out,'w') as f:
    f.write('\n'.join(ls))