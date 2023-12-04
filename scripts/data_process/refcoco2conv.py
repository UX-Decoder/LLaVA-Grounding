import json
refcoco_file="/comp_robot/cv_public_dataset/coco/annotations/grounding_train2017.json"
with open(refcoco_file) as f:
    refcoco=json.load(f)
instructions=[
"Could you offer a concise description of the object % ?",
"Please give a short summary of the object % .",
"Kindly provide a succinct description of the object % .",
"I'd appreciate a brief overview of the object % .",
"Could you summarize the object %  in a few words?",
"Please provide a brief explanation of the object % .",
"I'd like a quick description of the object % , please.",
"Could you quickly describe the object % ?",
"Please offer a condensed description of the object % .",
"Can you briefly characterize the object % ?",
"Kindly provide a brief description of the object % .",
"Would you mind giving a brief account of the object % ?",
"Can you sketch a short description of the object % ?",
"Please deliver a terse description of the object % .",
"Could you delineate the object %  briefly?",
"May I request a compact description of the object % ?",
"Could you provide a short characterization of the object % ?",
"Please offer a quick sketch of the object % .",
"Can you encapsulate the object's % details briefly?",
"I'd appreciate a rapid description of the object % .",
"Please present a succinct summary of the object % .",
"Would you briefly outline the object's % features?",
"Can you give a quick rundown of the object % ?",
"Please furnish a concise portrayal of the object % .",
"Could you briefly illustrate the main aspects of the object % ?",
"May I have a short exposition on the object % ?",
"Would you be able to give a brief depiction of the object % ?",
"Can you quickly highlight the key points of the object % ?",
"I'd like a brief elaboration on the object % , please.",
"Could you offer a quick snapshot of the object % ?",
"Please provide a brief account of the object's % characteristics.",
]

# format of output: [{"id": "000000033471", "image": "000000033471.jpg", "conversations": [{"from": "human", "value": "<image>\nWhat are the colors of the object <obj> in the image?"}, {"from": "gpt", "value": "The bus in the image is white and red."},],"q_gd_ls": [[366501]]},]
output_list=[]
import random
for ann in refcoco['annotations']:
    id=str(ann['image_id']).zfill(12)
    image=id+'.jpg'
    conversations=[]
    answers=[sent['raw'] for sent in ann['sentences']]
    # for answer in answers:
    instruct=random.choice(instructions)
    conversations.append({"from": "human", "value": instruct})
    conversations.append({"from": "gpt", "value": answers})
    q_gd_ls=[[ann['ann_id']]]
    output_list.append({"id": id, "image": image, "conversations": conversations,"q_gd_ls": q_gd_ls})

output_file="/comp_robot/cv_public_dataset/coco/annotations/grounding_train2017_instruct.json"
with open(output_file, 'w') as f:
    json.dump(output_list, f)