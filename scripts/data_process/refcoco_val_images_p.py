import json
coco_json_tr='/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017.json'
coco_json_val='/comp_robot/cv_public_dataset/coco/annotations/panoptic_val2017.json'
with open(coco_json_tr,'r') as f:
    coco_data_tr=json.load(f)
with open(coco_json_val,'r') as f:
    coco_data_val=json.load(f)

refcoco_json='/comp_robot/cv_public_dataset/coco/annotations/refcocop_unc.json'
with open(refcoco_json,'r') as f:
    refcoco_data=json.load(f)
im2anns={}
for ann in refcoco_data['annotations']:
    image_id=ann['image_id']
    if image_id not in im2anns:
        im2anns[image_id]=[]
    im2anns[image_id].append(ann)
assert len(im2anns.keys())==len(refcoco_data['images'])
new_images=[]
new_annos=[]
for image in (coco_data_tr['images']+coco_data_val['images']):
    image_id=image['id']
    if image_id in im2anns:
        new_images.append(image)

for ann in (coco_data_tr['annotations']+coco_data_val['annotations']):
    image_id=ann['image_id']
    if image_id in im2anns:
        new_annos.append(ann)

assert len(new_images)==len(new_annos)
assert len(new_images)==len(im2anns.keys())

coco_data_tr['images']=new_images
coco_data_tr['annotations']=new_annos
with open('/comp_robot/cv_public_dataset/coco/annotations/panoptic_refcocop_unc.json','w') as f:
    json.dump(coco_data_tr,f)

