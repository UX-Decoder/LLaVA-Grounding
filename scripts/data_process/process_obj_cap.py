import json
import os
import tqdm
data_dir="/comp_robot/cv_public_dataset/paco/annotations"
out_file="/comp_robot/cv_public_dataset/paco/annotations/paco_lvis_v1_train_obj_desc_merged_st2.json"
annos=[]
anno_ids=set()
file_list=os.listdir(data_dir)
file_list=[file_name for file_name in file_list if 'obj_desc' in file_name]
for file_name in tqdm.tqdm(file_list):
    with open(os.path.join(data_dir,file_name)) as f:
        data_tmp=json.load(f)
    for anno in data_tmp['annotations']:
        if anno['category_id']<2000 and 'description' in anno.keys() and 'obj_description' in anno.keys() and anno['id'] not in anno_ids:
            annos.append(anno)
            anno_ids.add(anno['id'])
data_tmp['annotations']=annos
with open(out_file,'w') as f:
    json.dump(data_tmp,f)

# with open(out_file) as f:
#     data_tmp=json.load(f)
print(len(data_tmp['annotations']))
print(len(data_tmp['images']))