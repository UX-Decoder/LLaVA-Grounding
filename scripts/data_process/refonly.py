import json
pano_path='/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017.json'
with open(pano_path) as f:
    pano_data=json.load(f)

ref_path='/comp_robot/cv_public_dataset/coco/annotations/grounding_train2017.json'
with open(ref_path) as f:
    ref_data=json.load(f)

new_pano_img=[]
new_pano_ann=[]
ref_image_set=set()
for ann in ref_data['annotations']:
    ref_image_set.add(ann['image_id'])

for img in pano_data['images']:
    if img['id'] in ref_image_set:
        new_pano_img.append(img)

for ann in pano_data['annotations']:
    if ann['image_id'] in ref_image_set:
        new_pano_ann.append(ann)

pano_data['images']=new_pano_img
pano_data['annotations']=new_pano_ann
out_file='/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017_refonly.json'
with open(out_file,'w') as f:
    json.dump(pano_data,f)