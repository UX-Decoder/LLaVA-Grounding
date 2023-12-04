#Ours=/comp_robot/lihongyang/code/VLLMs/LLaVA_new/llava_bench_results/llava_stage2_new_joint_seg0.1_data_v2_detail_checkpoint-6000/ours_llava_bench_results_thre_0.25_cut.jsonl
#CogVLM_Original=/comp_robot/lihongyang/rentianhe/CogVLM/cogvlm_llava_bench_results_specific.jsonl
#CogVLM_GC=/comp_robot/lihongyang/rentianhe/CogVLM/cogvlm_llava_bench_results_specific_prompt.jsonl
#Shikra_Original=/comp_robot/lihongyang/code/VLLMs/shikra/llava_bench_results/shikra_text_box_results.jsonl
#Shikra_GC=/comp_robot/lihongyang/code/VLLMs/shikra/llava_bench_results/shikra_text_box_results_SpotCap.jsonl
#MiniGPTv2=/comp_robot/lihongyang/rentianhe/MiniGPT-4/minigptv2_llava_bench_results.jsonl
CogVLM_Original_fix=/comp_robot/lihongyang/rentianhe/CogVLM/cogvlm_llava_bench_results_90qa_fixQ.jsonl
CogVLM_GC_fix=/comp_robot/lihongyang/rentianhe/CogVLM/cogvlm_llava_bench_results_90qa_fixQ_specific_prompt.jsonl
#bash scripts/eval.sh $Ours ours
#bash scripts/eval.sh $CogVLM_Original cogvlm_original
#bash scripts/eval.sh $CogVLM_GC cogvlm_gc
#bash scripts/eval.sh $Shikra_Original shikra_original
#bash scripts/eval.sh $Shikra_GC shikra_gc
#bash scripts/eval.sh $MiniGPTv2 minigptv2
bash scripts/eval.sh $CogVLM_Original_fix cogvlm_original_fix
bash scripts/eval.sh $CogVLM_GC_fix cogvlm_gc_fix