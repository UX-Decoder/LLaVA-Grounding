import json
import os
import sys
with_dir1=sys.argv[1]
with_non_gd=sys.argv[2]

instruct_dir1='/comp_robot/zhanghao/datasets/llava/instruct150k_brackets'
instruct_dir2='/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1'
data_no_ground='/comp_robot/cv_public_dataset/CC12M_zh/LLaVA-CC3M-Pretrain-595K/llava_instruct_150k.json'
merged_file='merged.json'
instruct_dir2_file_list=os.listdir(instruct_dir2)
instruct_dir2_file_list=[f for f in instruct_dir2_file_list if f.endswith('.json') and 'merged' not in f]
instruct_dir1_file_list=os.listdir(instruct_dir1)
instruct_dir1_file_list=[f for f in instruct_dir1_file_list if f.endswith('.json') and 'merged' not in f]
results=[]
if with_dir1=='1':
    print('with dir1')
for f in instruct_dir1_file_list:
    # import pdb;pdb.set_trace()
    if f not in instruct_dir2_file_list:
        if with_dir1=='1':
            with open(os.path.join(instruct_dir1,f)) as f:
                results.extend(json.load(f))
    else:
        with open(os.path.join(instruct_dir2, f)) as f:
            results.extend(json.load(f))
if with_non_gd=='1':
    print('with non gd')
    res_non_gd=json.load(open(data_no_ground, 'r'))
    results.extend(res_non_gd)
with open(os.path.join(instruct_dir2,merged_file),'w') as f:
    json.dump(results,f,indent=4)