LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models
========


<!-- [[`Paper`](xxx)] [[`BibTex`](#black_nib-citation)] -->

## :fire: News
[2023/12/5] Our paper is available in arxiv.


## Contents
### Install
1. Clone this repository and navigate to LLaVA-Grounding fold:
```shell
git clone https://github.com/UX-Decoder/LLaVA-Grounding.git
cd LLaVA-Grounding
```
2. Install required packages:
```
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
### LLaVA-Grounding Weights
Please check out our [Model Zoo](https://github.com/UX-Decoder/LLaVA-Grounding/blob/main/docs/MODEL_ZOO.md) for all public LLaVA-Grounding checkpoints, and the instructions on how to use the weights.
### Demo
After downloading model weights, simply conduct the following commends to run demo on your own machine.
```shell
CUDA_VISIBLE_DEVICES=0 python gradio_demo/LLaVA_G_Demo.py --path_vision_cfg path_to_vision_cfg --path_inter_cfg path_to_inter_cfg --model_path path_to_ckpt_dir

# for example, after downloading weights into checkpoints/llava_grounding
CUDA_VISIBLE_DEVICES=0 python gradio_demo/LLaVA_G_Demo.py --path_vision_cfg configs/openseed/openseed_swint_lang_joint_2st_v2_data_end_with_interaction.yaml --path_inter_cfg configs/semsam/idino_swint_1_part_data_llm_ref_feat_all_16_det_pretrainv1.yaml --model_path checkpoints/llava_grounding
```