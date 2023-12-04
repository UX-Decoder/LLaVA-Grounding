import json
ori_data_path="/comp_robot/zhanghao/detail_23k.json"
data_path="/comp_robot/zhanghao/datasets/llava/detailed23k_brackets_out/detailed23k.json"
succ_path="/comp_robot/zhanghao/datasets/llava/detailed23k_brackets_out/detailed23k_succ.json"
detailed23k=json.load(open(ori_data_path, 'r'))
detailed23k_labeled=json.load(open(data_path, 'r'))
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
    return True
succ_ls=[]
fail_ls=[]
for data,ori_data in zip(detailed23k_labeled,detailed23k):
    if decide_succ(data):
        succ_ls.append(data)
    else:
        fail_ls.append(ori_data)
print("succ:",len(succ_ls))
print('fail:',len(fail_ls))
fail_split1="detailed23k_fail1.json"
fail_split2="detailed23k_fail2.json"
json.dump(succ_ls,open(succ_path,'w'),indent=4)
print(len(fail_ls)//2)
json.dump(fail_ls[:len(fail_ls)//2],open(fail_split1,'w'),indent=4)
json.dump(fail_ls[len(fail_ls)//2:],open(fail_split2,'w'),indent=4)