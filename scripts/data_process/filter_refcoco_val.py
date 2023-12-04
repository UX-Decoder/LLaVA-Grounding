import json
coco_json_tr='/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017_filter.json'
with open(coco_json_tr,'r') as f:
    coco_data_tr=json.load(f)
refcoco_json='/comp_robot/cv_public_dataset/coco/annotations/refcoco_unc.json'
with open(refcoco_json,'r') as f:
    refcoco_data=json.load(f)

refcocog_json='/comp_robot/cv_public_dataset/coco/annotations/refcocog_umd_val.json'
with open(refcocog_json,'r') as f:
    refcocog_data=json.load(f)

refcocop_json='/comp_robot/cv_public_dataset/coco/annotations/refcocop_unc.json'
with open(refcocop_json,'r') as f:
    refcocop_data=json.load(f)

ref_image_set=set()
for ann in refcoco_data['annotations']:
    ref_image_set.add(ann['image_id'])
for ann in refcocog_data['annotations']:
    ref_image_set.add(ann['image_id'])
for ann in refcocop_data['annotations']:
    ref_image_set.add(ann['image_id'])

filtered_images=[]
filtered_annos=[]
for image in coco_data_tr['images']:
    image_id=image['id']
    if image_id not in ref_image_set:
        filtered_images.append(image)
for ann in coco_data_tr['annotations']:
    image_id=ann['image_id']
    if image_id not in ref_image_set:
        filtered_annos.append(ann)

assert len(filtered_images)==len(filtered_annos)
coco_data_tr['images']=filtered_images
coco_data_tr['annotations']=filtered_annos
with open('/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017_filtered_ref.json','w') as f:
    json.dump(coco_data_tr,f)
