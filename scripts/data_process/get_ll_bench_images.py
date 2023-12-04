import json
question_file="playground/data/coco2014_val_qa_eval/qa90_questions.jsonl"
im_path="/comp_robot/cv_public_dataset/coco/val2014"
out_im_dir="question_ims"
import os
os.makedirs(out_im_dir,exist_ok=True)
for line in open(question_file):
    q=json.loads(line)
    im_file=q['image']
    os.system(f"cp {os.path.join(im_path,'COCO_val2014_'+im_file)} {os.path.join(out_im_dir,'COCO_val2014_'+im_file)}")
    