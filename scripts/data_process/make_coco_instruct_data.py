import json
# processed_instruct_data='/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged_end.json'
# processed_instruct_data_out='/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged_end_fixed.json'
# processed_instruct_data='/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged_v2.json'
# processed_instruct_data_out='/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged_v2_fixed.json'
processed_instruct_data='/comp_robot/zhanghao/datasets/llava/merged_v3.json'
processed_instruct_data_out='/comp_robot/zhanghao/datasets/llava/merged_v3_fixed.json'

with open(processed_instruct_data) as f:
    data=json.load(f)

image_set=set()

for d in data:
    image_set.add(int(d['id']))

# image_set=set(image_set)
coco_instance_data_train='/comp_robot/cv_public_dataset/coco/annotations/instances_train2014.json'
with open(coco_instance_data_train) as f:
    coco_data=json.load(f)
# coco_instance_data_val='/comp_robot/cv_public_dataset/coco/annotations/instances_val2017.json'
# with open(coco_instance_data_val) as f:
#     coco_data_val=json.load(f)
# coco_data['images'].extend(coco_data_val['images'])
# coco_data['annotations'].extend(coco_data_val['annotations'])
coco_data['images']=[d for d in coco_data['images'] if d['id'] in image_set]
coco_data['annotations']=[d for d in coco_data['annotations'] if d['image_id'] in image_set]
coco_data_set=set([d['id'] for d in coco_data['images']])
assert len(coco_data_set)==len(image_set)
import collections
imid2annid=collections.defaultdict(set)
for ann in coco_data['annotations']:
    imid2annid[ann['image_id']].add(ann['id'])

new_data=[]
for d in data:
    # image_set.add(int(d['id']))
    if 'gd_ls' not in d:
        print(d['id'])
        continue
    valid=True
    for gd in d['gd_ls']:
        if gd is not None:
            # import pdb;pdb.set_trace()
            try:
                for gd_ in gd:
                    assert gd_ in imid2annid[int(d['id'])]
            except Exception as e:
                print(d['id'])
                print(gd)
                valid=False
                break
    if valid:
        new_data.append(d)


# out_file='/comp_robot/cv_public_dataset/coco/annotations/instances_train2017_instruct_grounding_fix.json'
# out_file='/comp_robot/cv_public_dataset/coco/annotations/instances_train2017_instruct_grounding_fix_v2.json'
out_file='/comp_robot/cv_public_dataset/coco/annotations/instances_train2017_instruct_grounding_fix_v3.json'
with open(out_file,'w') as f:
    json.dump(coco_data,f)

with open(processed_instruct_data_out,'w') as f:
    json.dump(new_data,f)