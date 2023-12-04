import json
q_file="playground/data/coco2014_val_qa_eval/qa90_questions.jsonl"
val_anno_file="/comp_robot/cv_public_dataset/coco/annotations/instances_val2014.json"
q_image_set=set()
with open(q_file) as f:
    for line in f:
        q=json.loads(line)
        q_img_id=int(q['image'][:-4])
        q_image_set.add(q_img_id)

# add more images from val_anno_file to q_image_set until len(q_image_set)=1000
with open(val_anno_file) as f:
    val_anno=json.load(f)
    import random
    random.shuffle(val_anno['images'])
    for img in val_anno['images']:
        if len(q_image_set)==1000:
            break
        if img['id'] not in q_image_set:
            q_image_set.add(img['id'])

# write q_image_set to file
q_image_set_file="playground/data/coco2014_val_qa_eval/q_image_set.txt"
with open(q_image_set_file,'w') as f:
    for img_id in q_image_set:
        f.write(str(img_id)+'\n')