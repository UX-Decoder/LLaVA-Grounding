import json
detailed="/comp_robot/zhanghao/datasets/llava/detailed23k_brackets_out/detailed23k_succ_v2.json"
complex="/comp_robot/zhanghao/datasets/llava/complex_brackets_out/complex_succ_v2.json"
conv="/comp_robot/zhanghao/datasets/llava/conversation_58k_brackets_out/conversation_58k_succ_v2.json"
output="/comp_robot/zhanghao/datasets/llava/merged_v3.json"
detailed_data=json.load(open(detailed))
complex_data=json.load(open(complex))
conv_data=json.load(open(conv))
data_list=[]
data_list.extend(detailed_data)
data_list.extend(complex_data)
data_list.extend(conv_data)
json.dump(data_list,open(output,'w'),indent=4)