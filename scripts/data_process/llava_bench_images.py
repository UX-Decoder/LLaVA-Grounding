import json
# coco_train_file="/comp_robot/cv_public_dataset/coco/annotations/instances_train2017.json"
# coco_val_file="/comp_robot/cv_public_dataset/coco/annotations/instances_val2017.json"
# coco_train = json.load(open(coco_train_file))
# coco_val = json.load(open(coco_val_file))
# coco_train_val
llava_bench_file="/comp_robot/cv_public_dataset/coco/annotations/llava_bench_qa90_gpt4_conv_end.json"
llava_bench = json.load(open(llava_bench_file))
llava_bench_ids = list(set([x['id'] for x in llava_bench]))
llava_bench_ids=['*'+id+'*' for id in llava_bench_ids]
llava_bench_ids=' '.join(llava_bench_ids)
print(llava_bench_ids)
