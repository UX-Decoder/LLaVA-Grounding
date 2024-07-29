🌋 LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models
========

[[Project Page](https://llava-vl.github.io/llava-grounding)] [[Arxiv](https://arxiv.org/abs/2312.02949)]  [[Demo](https://llava-grounding.deepdataspace.com/
)]  [[Model Zoo](https://github.com/UX-Decoder/LLaVA-Grounding/blob/main/docs/MODEL_ZOO.md)] 
<!-- [[`Paper`](xxx)] [[`BibTex`](#black_nib-citation)] -->

## :fire: News
[2024/1/14] Our training code is released.

[2023/12/6] Our paper is available in arxiv.


## Contents
- [🌋 LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models](#-llava-grounding-grounded-visual-chat-with-large-multimodal-models)
  - [:fire: News](#fire-news)
  - [Contents](#contents)
    - [Install](#install)
    - [LLaVA-Grounding Weights](#llava-grounding-weights)
    - [Demo](#demo)
    - [Training data](#training-data)
      - [Flickr30k](#flickr30k)
      - [COCO](#coco)
      - [LLaVA](#llava)
    - [Training](#training)
    - [Citation](#citation)

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
4. Install packages necessary for [OpenSeeD](https://github.com/IDEA-Research/OpenSeeD) and [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM).

### LLaVA-Grounding Weights
Please check out our [Model Zoo](https://github.com/UX-Decoder/LLaVA-Grounding/blob/main/docs/MODEL_ZOO.md) for all public LLaVA-Grounding checkpoints, and the instructions on how to use the weights.
### Demo
After downloading model weights, simply conduct the following commends to run demo on your own machine.
```shell
CUDA_VISIBLE_DEVICES=0 python gradio_demo/LLaVA_G_Demo.py --path_vision_cfg path_to_vision_cfg --path_inter_cfg path_to_inter_cfg --model_path path_to_ckpt_dir

# for example, after downloading weights into checkpoints/llava_grounding
CUDA_VISIBLE_DEVICES=0 python gradio_demo/LLaVA_G_Demo.py --path_vision_cfg configs/openseed/openseed_swint_lang_joint_2st_visual_prompt.yaml --path_inter_cfg configs/semsam/visual_prompt_encoder.yaml --model_path checkpoints/llava_grounding
```

Please refer to our [Online Demo](https://llava-grounding.deepdataspace.com/) for the more detailed user's guidence.
### Training data
```text
data
├── flickr30k_entities
│   ├── train/
│   ├── val/
│   ├── annotations
│          ├──final_flickr_separateGT_train.json
│          ├──final_flickr_separateGT_val.json
├── coco
│   ├── train2014/
│   ├── train2017/
│   ├── panoptic_train2017/
│   ├── panoptic_semseg_train2017/
│   ├── annotations
│   │      ├──instances_train2017.json
│   │      ├──instances_train2017_gvc.json
│   │      ├──grounded_visual_chat_data.json
│   │      ├──instances_train2014_filter.json
│   │      ├──panoptic_train2017_filter.json
│   │      ├──grounding_train2017.json
├── llava
│   ├── annotations
│          ├── cap600k_brackets_all.json
│          ├── llava_instruct_150k.json
│          ├── llava_instruct_150k_visual_prompt.json

```
#### Flickr30k
Please refer to [MDETR's pre-processed flickr30k data](https://github.com/ashkamath/mdetr/blob/main/.github/flickr.md).
#### COCO
Please download coco train2014 and train2017 images and panoptic segmentation and semantic segmentation data. Other annoations can be downloaded [here](https://github.com/UX-Decoder/LLaVA-Grounding/releases/tag/train_data).
#### LLaVA
The processed annotations can be downloaded [here](https://github.com/UX-Decoder/LLaVA-Grounding/releases/tag/train_data).
### Training
Stage 1
```shell
bash scripts/pretrain_joint.py
```
Stage 2
```shell
bash scripts/finetune.py
```
Stage 3
```shell
bash scripts/finetune_visual_prompt.py
```
### Citation
If you find LLaVA-Grounding useful for your research and applications, please cite using this BibTeX:
```bibtex

@misc{zhang2023llavagrounding,
      title={LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models},
      author={Hao Zhang and Hongyang Li and Feng Li and Tianhe Ren and Xueyan Zou and Shilong Liu and Shijia Huang and Jianfeng Gao and Lei Zhang and Chunyuan Li and Jianwei Yang},
      year={2023},
      booktitle={arXiv}
}

@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={arXiv:2304.08485},
      year={2023}
}
```
