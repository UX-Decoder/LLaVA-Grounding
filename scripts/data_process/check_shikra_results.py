import json
import os
result_dir="/comp_robot/zhanghao/model/gpt4eval_results_for_grounding/shikra_gc_90/"

for file in os.listdir(result_dir):
    if file.endswith(".json"):
        with open(os.path.join(result_dir, file), "r") as f:
            data = json.load(f)
        print(data['num_positive_ref'],data['num_positive_obj'],file)