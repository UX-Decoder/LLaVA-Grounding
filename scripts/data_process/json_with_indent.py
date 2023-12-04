import json
input="/comp_robot/lihongyang/code/VLLMs/LLaVA_new/merged_v3_fixed_inter_marker.json"
output="./merged_v3_fixed_inter_marker_indent.json"
with open(input) as f:
    data = json.load(f)
with open(output,'w') as f:
    json.dump(data[:100],f,indent=4)