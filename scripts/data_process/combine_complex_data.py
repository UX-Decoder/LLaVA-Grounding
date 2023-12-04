import json
import os
data_dir="/comp_robot/zhanghao/datasets/llava/complex_brackets_out/"
data_list=[]
def decide_succ(data):
    if 'gd_ls' not in data:
        return False
    gd_ls=data['gd_ls']
    if len(gd_ls)==0:
        return False
    conv=data['conversations']
    question=conv[0]['value']
    if 'with grounding' not in question:
        return False
    answer=conv[1]['value']
    if 'Please provide' in answer:
        return False
    if answer.count('<g_s>')!=len(gd_ls):
        return False
    if answer.count('<g_e>')!=len(gd_ls):
        return False
    if answer.count('<seg>')!=len(gd_ls):
        return False
    if '<g_s> <seg>' in answer:
        return False
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
json.dump(fail_ls,open(os.path.join(data_dir,'complex_failed_v2.json'),'w'),indent=4)
json.dump(data_list,open(os.path.join(data_dir,'complex_succ_v2.json'),'w'),indent=4)