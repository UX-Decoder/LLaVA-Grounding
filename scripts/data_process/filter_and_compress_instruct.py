import json
import os
# from .utils import decide_succ
data_path="/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged.json"
data=json.load(open(data_path,'r'))
detailed_path="/comp_robot/zhanghao/datasets/llava/detailed23k_brackets_out/detailed23k_succ_merged.json"
def decide_succ(data_sample):
    if 'gd_ls' not in data_sample:
        return False
    gd_ls=[gd for gd in data_sample['gd_ls'] if gd is not None]
    if len(gd_ls)==0:
        return False
    conversations=data_sample["conversations"]
    gd_len=len(data_sample['gd_ls'])
    count_g_s=sum([c['value'].count('<g_s>') for c in conversations])
    count_g_e=sum([c['value'].count('<g_e>') for c in conversations])
    count_seg=sum([c['value'].count('<seg>') for c in conversations])
    if gd_len!=count_seg:
        return False
    if gd_len!=count_g_s:
        return False
    if gd_len!=count_g_e:
        return False
    return True


##########filter data########################################
succ_ls=[]
fail_ls=[]
for d in data:
    if decide_succ(d):
        succ_ls.append(d)
    else:
        fail_ls.append(d)
print("succ:",len(succ_ls))
print('fail:',len(fail_ls))
detailed_data=json.load(open(detailed_path,'r'))
import pdb;pdb.set_trace()
merged_data=succ_ls+detailed_data*3
import random
random.shuffle(merged_data)
output_path="/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged_v2.json"
with open(output_path,'w') as f:
    json.dump(merged_data,f,indent=4)
##########compress data########################################
# def compress_conversation(conversations,gd_ls):
#     new_conversations=[]
#     for i,gd in enumerate(gd_ls):
#         if gd is None:
#             for question, answer in zip(conv[::2], conv[1::2]):
#                 if "with grounding" not in question['value']:
#                     continue
#                 answer.split('<seg>')
#
#
#
# for data_per_image in data:
#     conv=data_per_image['conversations']
#     for question,answer in zip(conv[::2],conv[1::2]):
#
#     data_per_image['conversations']=conv