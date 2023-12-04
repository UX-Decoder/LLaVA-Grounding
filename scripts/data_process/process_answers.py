import json
# answer_file='/comp_robot/zhanghao/model/llava_stage2_new_gd_flickr_coco_only_w4_seg0.2/checkpoint-19000/answer_gd.json'
import sys
answer_file=sys.argv[1]
with open(answer_file) as f:
    answers = [json.loads(line) for line in f]
    for a in answers:
        a['text'] = a['text'].replace(' (with grounding)', '')
        a['text'] = a['text'].replace('<g_s> ', '')
        a['text'] = a['text'].replace(' <g_e> <seg>', '')
        a['text'] = a['text'].replace('<g_e>', '')
        a['text'] = a['text'].replace('<seg>', '')
        a['text'] = a['text'].replace('<g_s>', '')
answer_file_out=answer_file.replace('.json','_new.json')
with open(answer_file_out, 'w') as f:
    for a in answers:
        f.write(json.dumps(a) + '\n')