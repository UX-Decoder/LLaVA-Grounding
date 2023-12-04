import json
import os
data_dir="/comp_robot/zhanghao/datasets/llava/conversation_58k_brackets_out"
data_list=[]
def decide_succ(data):
    if 'gd_ls' not in data:
        return False
    gd_ls=data['gd_ls']
    if len(gd_ls)==0:
        return False
    conv=data['conversations']
    questions=[conv[i]['value'] for i in range(0,len(conv),2)]
    gd=False
    for question in questions:
        if 'with grounding' in question:
            gd=True
    if not gd:
        return False
    # if 'with grounding' not in question:
    #     return False
    answers=[conv[i]['value'] for i in range(1,len(conv),2)]
    c_s,c_e,c_g=0,0,0
    for answer in answers:
        if 'Please provide' in answer:
            return False
        if '<g_s> <seg>' in answer:
            return False
        c_s+=answer.count('<g_s>')
        c_e+=answer.count('<g_e>')
        c_g+=answer.count('<seg>')
    if c_s!=len(gd_ls):
        return False
    if c_e!=len(gd_ls):
        return False
    if c_g!=len(gd_ls):
        return False
    # if 'Please provide' in answer:
    #     return False
    # if answer.count('<g_s>')!=len(gd_ls):
    #     return False
    # if answer.count('<g_e>')!=len(gd_ls):
    #     return False
    # if answer.count('<seg>')!=len(gd_ls):
    #     return False

    return True

succ=0
fail=0
fail_ls=[]
import tqdm
for file in tqdm.tqdm(os.listdir(data_dir)):
    try:
        assert file.split('.')[1]=='json'
        image_id=int(file.split('.')[0])
    except:
        continue
    with open(os.path.join(data_dir,file),'r') as f:
        data=json.load(f)
    if decide_succ(data):
        data_list.append(data)
        succ+=1
    else:
        fail_ls.append(data)
        fail+=1
print(succ,fail)
json.dump(fail_ls,open(os.path.join(data_dir,'conversation_58k_failed_v2.json'),'w'),indent=4)
json.dump(data_list,open(os.path.join(data_dir,'conversation_58k_succ_v2.json'),'w'),indent=4)