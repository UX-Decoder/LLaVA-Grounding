import json
coco_json='/comp_robot/cv_public_dataset/coco/annotations/instances_train2014.json'

with open(coco_json,'r') as f:
    coco_data=json.load(f)

filter_images=[]
filter_annos=[]
img2anns={}

for ann in coco_data['annotations']:
    image_id=ann['image_id']
    if image_id not in img2anns:
        img2anns[image_id]=[]
    img2anns[image_id].append(ann)

for image in coco_data['images']:
    image_id=image['id']
    if image_id in img2anns:
        filter_images.append(image)
        # filter_annos+=img2anns[image_id]


# filter_images=[]
# for image in coco_data['images']:
#     image_id=image['id']
#     if image_id in filter_image_id:
#         filter_images.append(image)
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
# coco_data['annotations']=filter_annos
out_file='/comp_robot/cv_public_dataset/coco/annotations/instances_train2014_filter.json'
with open(out_file,'w') as f:
    json.dump(coco_data,f)
