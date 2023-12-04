import json
answer_file='/comp_robot/zhanghao/model/llava_stage2_new_gd_flickr_coco_only_w4_end_seg0.2/checkpoint-9000/answer_gd.json'
with open(answer_file) as f:
    answers = [json.loads(line) for line in f]
    for a in answers:
        # import pdb;pdb.set_trace()
        num_segs=a['text'].count('<seg>')
        if num_segs>0:
            a['text']=a['text'][0:a['text'].find('1:')-1]
        objs=[]
        for i in range(num_segs):
            start=a['text'].find(f'<g_s> {i+1} ')
            if start==-1:
                continue
            end=a['text'].find('<g_e>')
            objs.append(a['text'][start+len(f'<g_s> {i+1} '):end].lower().strip())
            a['text']=a['text'][:end-1]+a['text'][end+5:]
            a['text'] = a['text'].replace(f'<g_s> {i+1} ', '')

        summary='There are the following objects in the image. They are '+', '.join(objs)+'.'

        a['text'] = a['text'].replace('<g_s> ', '')
        # a['text'] = a['text'].replace(' <g_e> <seg>', '')
        a['text'] = a['text'].replace('<g_e>', '')
        a['text'] = a['text'].replace('<seg>', '')
        a['text'] = a['text'].replace('<g_s>', '')
        a['text']=a['text']+summary
answer_file_out=answer_file.replace('.json','_new_objs.json')
with open(answer_file_out, 'w') as f:
    for a in answers:
        f.write(json.dumps(a) + '\n')
