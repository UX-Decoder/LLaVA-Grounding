import json
instance_train="/comp_robot/cv_public_dataset/coco/annotations/instances_train2017.json"
with open(instance_train) as f:
    instance_train=json.load(f)
imageset1=set([ann['image_id'] for ann in instance_train['annotations']])
refcoco_instruct="/comp_robot/cv_public_dataset/coco/annotations/grounding_train2017_instruct.json"
with open(refcoco_instruct) as f:
    refcoco_instruct=json.load(f)

imageset2=set([int(conversation['id']) for conversation in refcoco_instruct])
new_imags=[]
new_anns=[]
for ann in instance_train['annotations']:
    if ann['image_id'] in imageset2:
        new_anns.append(ann)
        # new_imags.append(ann['image_id'])
for img in instance_train['images']:
    if img['id'] in imageset2:
        new_imags.append(img)

instance_train['annotations']=new_anns
instance_train['images']=new_imags
with open("/comp_robot/cv_public_dataset/coco/annotations/instances_train2017_refcoco.json", 'w') as f:
    json.dump(instance_train, f)