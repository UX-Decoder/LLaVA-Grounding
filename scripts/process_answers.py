import json
answer_file='/comp_robot/zhanghao/model/llava_stage2_new_gd/answer_gd.json'
with open(answer_file) as f:
    answers = [json.loads(line) for line in f]
    for a in answers:
        a['text'] = a['text'].replace(' (with grounding)', '')
        a['text'] = a['text'].replace('<g_s> ', '')
        a['text'] = a['text'].replace(' <g_e> <seg>', '')
        a['text'] = a['text'].replace('<g_e>', '')
        a['text'] = a['text'].replace('<seg>', '')
        a['text'] = a['text'].replace('<g_s>', '')
with open(answer_file, 'w') as f:
    for a in answers:
        f.write(json.dumps(a) + '\n')