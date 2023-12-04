import json
import os
import sys
key=sys.argv[1]
res_dir=f"/comp_robot/zhanghao/model/gpt4eval_results_for_grounding/{key}"
num_preds=0
num_gts=0
num_corrects_pred=0
num_corrects_gt=0
import tqdm
for file in tqdm.tqdm(os.listdir(res_dir)):
    if file.endswith('.json'):
        data=json.load(open(os.path.join(res_dir,file),'r'))
        if "num_positive_ref" not in data or  data["num_positive_ref"]>100 or data["num_positive_obj"]>100:
            print(file)
            data["num_positive_ref"]=0
            data["num_positive_obj"]=0
#        print(num_gts)
        if data['gt_boxes'].count(':')>100 or data['num_pred']>100:
            import pdb;pdb.set_trace()
        else:
            num_gts+=data['gt_boxes'].count(':')
            num_preds+=data['num_pred']
            # import pdb;pdb.set_trace()
            num_corrects_pred+=data["num_positive_ref"]
            num_corrects_gt+=data["num_positive_obj"]
recall=num_corrects_gt/num_gts
precision=num_corrects_pred/num_preds
f1=2*recall*precision/(recall+precision)
print("num_preds: %d, num_gts: %d, num_corrects_pred: %d, num_corrects_gt: %d"%(num_preds,num_gts,num_corrects_pred,num_corrects_gt))
print("recall: %f, precision: %f, f1: %f"%(recall,precision,f1))
