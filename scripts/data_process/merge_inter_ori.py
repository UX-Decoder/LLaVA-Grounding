import json
import os
inter_dir="/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_q"
merged_file="/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_q/merged.json"
inter_list=[]
file_list=os.listdir(inter_dir)
file_list=[file_name for file_name in file_list if 'json' in file_name and 'merged' not in file_name]


instruct="/comp_robot/cv_public_dataset/CC12M_zh/LLaVA-CC3M-Pretrain-595K/llava_instruct_150k.json"
instruct_out="/comp_robot/cv_public_dataset/CC12M_zh/LLaVA-CC3M-Pretrain-595K/llava_instruct_150k_instr.json"
with open(instruct) as f:
    data=json.load(f)

imgid2instr={}
for anno in data:
    if anno['id'] in imgid2instr:
        imgid2instr[anno['id']].append(anno)
    else:
        imgid2instr[anno['id']]=[anno]

from difflib import SequenceMatcher
succ_match=0
inter_list_out=[]
for file_name in file_list:
    with open(os.path.join(inter_dir,file_name)) as f:
        data_tmp=json.load(f)
    for anno in data_tmp:
        if 'new_qs' not in anno or anno['new_qs'] is None:
            continue
        # convs=anno['conversations']
        num_obj=sum([q.count('<obj>') for q in anno['new_qs'] if q is not None])
        if num_obj==0:
            continue
        num_gts=sum([len(gd) for gd in anno['q_gd_ls'] if gd is not None])
        assert num_obj==num_gts, 'num_obj: {}, num_gts: {}'.format(num_obj,num_gts)
        assert len(anno['new_qs'])==len(anno['q_gd_ls'])
        convs_=anno['conversations']
        assert len(convs_)//2==len(anno['new_qs'])
        inter_list.append(anno)
        img_id=anno['id']
        #match with instruct
        q0=convs_[0]['value']
            # convs)[0]['value']
        if img_id in imgid2instr:
            match_scores=[]
            for anno_instruct in imgid2instr[img_id]:
                convs=anno_instruct['conversations']
                match_scores.append(SequenceMatcher(None,q0,convs[0]['value']).ratio())
            max_score=max(match_scores)
            max_idx=match_scores.index(max_score)
            if max_score>0.6:
                imgid2instr[img_id][max_idx]['new_qs']=anno['new_qs']
                imgid2instr[img_id][max_idx]['q_gd_ls']=anno['q_gd_ls']
                convs_res= imgid2instr[img_id][max_idx]['conversations']
                for ii,(q_res,q_new) in enumerate(zip(convs_res[::2],anno['new_qs'])):
                    if ii==0:
                        pre_str='<image>\n'
                    else:
                        pre_str=''
                    if q_new is not None:
                        q_res['value']=pre_str+q_new
                succ_match+=1
                inter_list_out.append(imgid2instr[img_id][max_idx])

print(succ_match)
with open(merged_file,'w') as f:
    json.dump(inter_list,f)
import pdb;pdb.set_trace()
with open(instruct_out,'w') as f:
    json.dump(inter_list_out,f)