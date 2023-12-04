import json
coco_json='/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017.json'

with open(coco_json,'r') as f:
    coco_data=json.load(f)

filter_image_id=[]
filter_annos=[]
for ann in coco_data['annotations']:
    if ann['segments_info']==[]:
        print(ann)
        print('-----------------')
    else:
        filter_annos.append(ann)
        filter_image_id.append(ann['image_id'])

filter_images=[]
for image in coco_data['images']:
    image_id=image['id']
    if image_id in filter_image_id:
        filter_images.append(image)
# image2anns={}
# for ann in coco_data['annotations']:
#     image_id=ann['image_id']
#     if image_id not in image2anns:
#         image2anns[image_id]=[]
#     image2anns[image_id].append(ann)

# filter_images=[]
# for image in coco_data['images']:
#     image_id=image['id']
#     if image_id not in image2anns:
#         print(image_id)
#         print(image)
#         print('-----------------')
#     else:
#         filter_images.append(image)
#
coco_data['images']=filter_images
coco_data['annotations']=filter_annos
with open('/comp_robot/cv_public_dataset/coco/annotations/panoptic_train2017_filter.json','w') as f:
    json.dump(coco_data,f)
