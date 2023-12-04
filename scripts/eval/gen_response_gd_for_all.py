import os
import sys
start=int(sys.argv[1])*1000
end=int(sys.argv[2])*1000
root=sys.argv[3] #/comp_robot/zhanghao/model/llava_stage2_new_joint_seg0.2/
for i in range(start,end,1000):
    command = f"bash scripts/gen_response_gd.sh {root}/checkpoint-{i}/"
    os.system(command)