import json
# interleave_file="/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged.json"
# end_file="/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged_end.json"
# interleave_file="/comp_robot/zhanghao/code/all/LLaVA_0505/playground/data/coco2014_val_qa_eval/qa90_gpt4_conv.json"
# end_file="/comp_robot/zhanghao/code/all/LLaVA_0505/playground/data/coco2014_val_qa_eval/qa90_gpt4_conv_end.json"
# interleave_file="/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged_v2_fixed.json"
# end_file="/comp_robot/zhanghao/datasets/llava/instruct150k_brackets_out_1/merged_end_v2_fixed.json"
interleave_file="/comp_robot/zhanghao/datasets/llava/detailed23k_brackets_out/detailed23k_succ_merged.json"
end_file="/comp_robot/zhanghao/datasets/llava/detailed23k_brackets_out/detailed23k_succ_merged_end.json"
with open(interleave_file, "r") as f:
    data = json.load(f)


def add_numbers_to_substring(input_string, substring):
    count = 1
    index = 0
    result = []

    while index < len(input_string):
        next_index = input_string.find(substring, index)
        if next_index == -1:
            break

        result.append(input_string[index:next_index])
        result.append(f"{substring}{count} ")
        count += 1
        index = next_index + len(substring)

    result.append(input_string[index:])

    return ''.join(result)

new_data=[]
count_gd=0
count_non_gd=0
for iter,data_sample in enumerate(data):
    conversations=data_sample["conversations"]
    if 'gd_ls' not in data_sample:
        count_non_gd+=len(conversations)//2
        continue
    gd_ls=[gd for gd in data_sample['gd_ls'] if gd is not None]
    if len(gd_ls)==0:
        print('gd_ls error')
        count_non_gd += len(conversations) // 2
        continue
    gd_len=len(data_sample['gd_ls'])
    count_g_s=sum([c['value'].count('<g_s>') for c in conversations])
    count_g_e=sum([c['value'].count('<g_e>') for c in conversations])
    count_seg=sum([c['value'].count('<seg>') for c in conversations])
    if gd_len!=count_seg:
        print('seg error')
        # print()
        count_non_gd += len(conversations) // 2
        continue
    if gd_len!=count_g_s:
        print('gs error')
        print(iter)
        count_non_gd += len(conversations) // 2
        continue
    if gd_len!=count_g_e:
        print('ge error')
        count_non_gd += len(conversations) // 2
        continue


    num_rounds=len(conversations)//2
    for i in range(num_rounds):
        question=conversations[2*i]["value"]
        if 'with grounding' not in question:
            count_non_gd+=1
            continue
        answer=conversations[2*i+1]["value"]
        answer=add_numbers_to_substring(answer, "<g_s>")
        answer=answer.replace("<seg>","")
        tail=[f'{i+1}: <seg>' for i in range(answer.count("<g_s>"))]
        tail='; '.join(tail)
        answer+=f' {tail}.'
        conversations[2*i+1]["value"]=answer
        count_gd+=1
    new_data.append(data_sample)

print(f'count_gd:{count_gd}, count_non_gd:{count_non_gd}')
with open(end_file, "w") as f:
    json.dump(new_data, f)

