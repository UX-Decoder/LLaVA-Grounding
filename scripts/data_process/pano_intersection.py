import json
data1_f="/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017_refonly.json"
data1_f_out="/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017_refonly_filtered.json"
data2_f="/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017_filtered_ref.json"
with open(data1_f) as f:
    data1=json.load(f)
with open(data2_f) as f:
    data2=json.load(f)

data1_images=data1['images']
data2_images=data2['images']
data1_images_id=[i['id'] for i in data1_images]
data2_images_id=[i['id'] for i in data2_images]
data1_images_id_set=set(data1_images_id)
data2_images_id_set=set(data2_images_id)
intersection=data1_images_id_set.intersection(data2_images_id_set)

new_images=[]
for i in data1_images:
    if i['id'] in intersection:
        new_images.append(i)

new_annos=[]
for anno in data1['annotations']:
    if anno['image_id'] in intersection:
        new_annos.append(anno)

new_data={}
new_data['images']=new_images
new_data['annotations']=new_annos
data1.update(new_data)
with open(data1_f_out,'w') as f:
    json.dump(data1,f)
