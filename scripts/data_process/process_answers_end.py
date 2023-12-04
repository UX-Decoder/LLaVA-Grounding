import json
import sys
answer_file=sys.argv[1]
# answer_file='/comp_robot/zhanghao/model/llava_stage2_new_gd_flickr_coco_only_w4_end_seg0.2/checkpoint-9000/answer_gd.json'
with open(answer_file) as f:
    answers = [json.loads(line) for line in f]
    for a in answers:
        # import pdb;pdb.set_trace()
        num_segs=a['text'].count('<seg>')
        if num_segs>0:
            a['text']=a['text'][0:a['text'].find('1:')-1]
        for i in range(num_segs):
            a['text'] = a['text'].replace(f'<g_s> {i+1} ', '')
        #     if i!=num_segs-1:
        #         a['text'] = a['text'].replace(f'{i+1}: <seg> ; ', '')
        #     else:
        #         a['text'] = a['text'].replace(f'{i+1}: <seg> .', '')

        # a['text'] = a['text'].replace(' (with grounding)', '')
        a['text'] = a['text'].replace('<g_s> ', '')
        # a['text'] = a['text'].replace(' <g_e> <seg>', '')
        a['text'] = a['text'].replace('<g_e>', '')
        a['text'] = a['text'].replace('<seg>', '')
        a['text'] = a['text'].replace('<g_s>', '')
        a['text'] = ' '.join(a['text'].split())
answer_file_out=answer_file.replace('.json','_new.json')
with open(answer_file_out, 'w') as f:
    for a in answers:
        f.write(json.dumps(a) + '\n')
