import json
vg_train="/comp_robot/cv_public_dataset/goldg/vg_mdetr/annotations/final_vg_train.json"
with open(vg_train) as f:
    vg_train_data = json.load(f)

for image in vg_train_data['images']:
    assert len(image['tokens_negative'])==len(image['caption'].split('. '))