import json
answer_file="/comp_robot/zhanghao/code/all/LLaVA_0505/playground/data/coco2014_val_qa_eval/qa90_gpt4_answer_v2.jsonl"
question_file="/comp_robot/zhanghao/code/all/LLaVA_0505/playground/data/coco2014_val_qa_eval/qa90_questions.jsonl"
out_file="/comp_robot/zhanghao/code/all/LLaVA_0505/playground/data/coco2014_val_qa_eval/qa90_gpt4_conv.json"
results=[]
#id, image, conversations, gd_ls
for line_q,line_a in zip(open(question_file,'r').readlines(),open(answer_file,'r').readlines()):
    q=json.loads(line_q)
    a=json.loads(line_a)
    if q['question_id']!=a['question_id']:
        import pdb;pdb.set_trace()
    image_id=q['image']
    id=image_id[:-4]
    conversations=[{'from':'human','value':q['text']+" (with grounding)"},{'from':'gpt','value':a['new_answer']}]
    gd_ls=a['gd_ls']
    results.append({'id':id,'image':image_id,'conversations':conversations,'gd_ls':gd_ls})

with open(out_file,'w') as f:
    json.dump(results,f,indent=4)