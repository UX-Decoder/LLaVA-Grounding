import os
import cv2
import json
import torch
import collections
import transformers
import numpy as np
from llava.model import *
from typing import Dict
from llava import conversation as conversation_lib
from tqdm import tqdm
from detectron2.utils.file_io import PathManager
from llava.mm_utils import tokenizer_image_token
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN
from llava.eval.llava_mapper import COCOInstanceNewBaselineDatasetMapper as LLAVAInstanceNewBaselineDatasetMapper
grounding_start="<g_s>"
grounding_end="<g_e>"
SEG_TOKEN="<seg>"
BOX_TOKEN="#B#"
MARKER_TOKEN="#M#"

def load_jsonl_file(path_jsonl):
        import jsonlines
        data = []
        with jsonlines.open(path_jsonl, "r") as reader:
            for obj in reader:
                data.append(obj)
        return data
def save_jsonl_file(data, path_save):
    import jsonlines
    with jsonlines.open(path_save, "w") as writer:
        for item in data: 
            writer.write(item) 
def load_benchmark(image_root, path_benchmark):

    data = load_jsonl_file(path_benchmark)
    ret = []
    for d in data:
        image_name = d["image"]
        image_id = int(image_name.split(".")[0])
        image_file = os.path.join(image_root, "COCO_val2014_" + image_name)
        # conv = d["conversations"]
        conv = [
        {
            "from": "human", 
            "value": d["text"]
        },
        {
            "from": "gpt", 
            "value": "Placeholder."
        }
        ]
        conv[0]["value"] = DEFAULT_IMAGE_TOKEN + " " + conv[0]["value"] + " (with grounding)"
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                # "grounding_info": None,
                "conversations": [[conv, None]],
                "question_id": d["question_id"]
            }
        )
    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret
def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conv_prompt = conv.get_prompt()
        conv_prompt = conv_prompt.split("ASSISTANT: ")[0] + "ASSISTANT:"
        conversations.append(conv_prompt)

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class Evaluator_MM:
    def __init__(self, 
                 model_path, 
                 path_vision_model_cfg=None):
        model_paths = model_path.split("/")
        if model_paths[-1].startswith('checkpoint-'):
            self.model_name = model_paths[-2] + "_" + model_paths[-1]
        else:
            self.model_name = model_paths[-1]
        print("1. Constructing model...")
        self.tokenizer, self.model, self.image_processor, self.context_len = self.construct_model(
            model_path=model_path,
            model_name=self.model_name,
        )
        print("   Continue...")
        self.construct_vision_model(path_vision_model_cfg)
        print("Done.")
        self.image_processor=self.model.get_vision_tower().image_processor
        print("2. Loading Parameters...")
        self.load_parameters(model_path)
        print("Done.")
        self.model.eval()

        conversation_lib.default_conversation = conversation_lib.conv_templates["v1"]
        self.data_mapper  = LLAVAInstanceNewBaselineDatasetMapper(self.cfg_vision_model, False, tokenizer=self.tokenizer, image_processor=self.image_processor, preprocess=preprocess_v1)
    
    def construct_model(self, model_path, model_base=None, model_name=None, load_8bit=False, load_4bit=False, device_map="auto"):
        import os
        import shutil
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
        from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

        kwargs = {"device_map": device_map}

        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        if 'llava' in model_name.lower():
            # Load LLaVA model
            if 'lora' in model_name.lower() and model_base is not None:
                lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print('Loading LLaVA from base model...')
                model = LlavaLlamaForCausalLM_gd.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
                if model.lm_head.weight.shape[0] != token_num:
                    model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                    model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

                print('Loading additional LLaVA weights...')
                if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                else:
                    # this is probably from HF Hub
                    from huggingface_hub import hf_hub_download
                    def load_from_hf(repo_id, filename, subfolder=None):
                        cache_file = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            subfolder=subfolder)
                        return torch.load(cache_file, map_location='cpu')
                    non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                model.load_state_dict(non_lora_trainables, strict=False)

                from peft import PeftModel
                print('Loading LoRA weights...')
                model = PeftModel.from_pretrained(model, model_path)
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
                print('Model is loaded...')
            elif model_base is not None:
                # this may be mm projector only
                print('Loading LLaVA from base model...')
                if 'mpt' in model_name.lower():
                    if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                        shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    model = LlavaLlamaForCausalLM_gd.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM_gd.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            else:
                if 'mpt' in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = LlavaLlamaForCausalLM_gd.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = LlavaLlamaForCausalLM_gd.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        else:
            # Load language model
            if model_base is not None:
                # PEFT model
                from peft import PeftModel
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
                print(f"Loading LoRA weights from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
                print(f"Merging weights")
                model = model.merge_and_unload()
                print('Convert to FP16...')
                model.to(torch.float16)
            else:
                use_fast = False
                if 'mpt' in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

        image_processor = None

        if 'llava' in model_name.lower():
            mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device='cuda', dtype=torch.float16)
            image_processor = vision_tower.image_processor

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        return tokenizer, model, image_processor, context_len
    def construct_vision_model(self, path_vision_model_cfg):
        from detectron2.config import LazyConfig
        from llava.model.openseed import build_model
        from llava.model.openseed.BaseModel import BaseModel

        def get_config_from_name(cfg, dataset_name="flickr"):
            # adjust config according to dataset, flickr by default
            if 'sam' in dataset_name:
                cfg.update(cfg['SAM'])
                return cfg
            elif 'flickr' in dataset_name:
                cfg.update(cfg['flickr'])
                return cfg
            elif 'coco_instruct_train' in dataset_name:
                cfg.update(cfg['coco_instruct'])
                return cfg
            elif 'lisa' in dataset_name:
                cfg.update(cfg['LISA_REF'])
                return cfg
            elif 'llava' in dataset_name:
                cfg.update(cfg['llava'])
                return cfg
            elif 'vg' in dataset_name:
                cfg.update(cfg['vg'])
                return cfg
            elif 'part' in dataset_name and 'pascal_part' not in dataset_name and 'partimagenet' not in dataset_name:
                cfg.update(cfg['part'])
                return cfg
            elif 'pascal' in dataset_name or 'paco' in dataset_name or 'partimagenet' in dataset_name :
                cfg.update(cfg['PSACAL_PART'])
                return cfg
            elif 'coco' in dataset_name and 'refonly' in dataset_name:
                # if 'COCO' in cfg.keys():
                cfg.update(cfg['COCO_REF'])
                return cfg
            elif 'refcoco' in dataset_name or "flickr_val" in dataset_name:
                cfg.update(cfg['REF'])
                return cfg
            elif 'coco' in dataset_name:
                if 'COCO' in cfg.keys():
                    cfg.update(cfg['COCO'])
                return cfg
            elif "mapillary" in dataset_name:
                if 'MAPILLARY' in cfg.keys():
                    cfg.update(cfg['MAPILLARY'])
                return cfg
            elif 'ade' in dataset_name:
                if 'ADE20K' in cfg.keys():
                    cfg.update(cfg['ADE20K'])
                return cfg
            elif 'imagenet' in dataset_name:
                if 'IMAGENET' in cfg.keys():
                    cfg.update(cfg['IMAGENET'])
                return cfg
            elif 'vlp' in dataset_name:
                cfg.update(cfg['VLP'])
                return cfg
            elif 'sun' in dataset_name:
                cfg.update(cfg['SUN'])
                return cfg
            elif 'object365' in dataset_name:
                cfg.update(cfg['OBJECT365'])
                return cfg
            elif 'scan' in dataset_name:
                cfg.update(cfg['SCAN'])
                return cfg
            elif 'cityscape' in dataset_name:
                cfg.update(cfg['CITY'])
                return cfg
            elif 'bdd' in dataset_name:
                cfg.update(cfg['BDD'])
                return cfg
            else:
                assert False, "dataset not support."
        self.cfg_vision_model = LazyConfig.load(path_vision_model_cfg)
        vision_model = BaseModel(self.cfg_vision_model, build_model(self.cfg_vision_model))
        vision_model.eval()
        self.model.seg_model = vision_model
        self.model.seg_model.model = self.model.seg_model.model.to(self.model.device)
        # print("Configuring for Dataset Mapper ...")
        self.cfg_vision_model = get_config_from_name(self.cfg_vision_model)
    
    def load_parameters(self, path_model):
        print("Loading Whole Model ...")
        loaded_dict = dict()
        for model_file in os.listdir(path_model):
            if model_file.endswith('.bin') and model_file.startswith('pytorch_model'):
                loaded_dict.update(torch.load(os.path.join(path_model, model_file), map_location='cpu'))
        self.model.load_state_dict(loaded_dict, strict=True)
    @torch.inference_mode()
    def evaluate_sample(self, input_data, get_box=True, get_mask=False):
        text, boxes, masks = self.model.forward_eval(input_data)
        returns = [text,]
        if get_box:
            returns.append(boxes)
        if get_mask:
            returns.append(masks)
        
        return returns

class Evaluator_MM_Inter(Evaluator_MM):
    def __init__(self, model_path, path_vision_model_cfg=None, path_inter_model_cfg=None):
        self.path_inter_model_cfg = path_inter_model_cfg
        super().__init__(model_path, path_vision_model_cfg)
    def construct_model(self, model_path, model_base=None, model_name=None, load_8bit=False, load_4bit=False, device_map="auto"):
        import os
        import shutil
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
        from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        kwargs = {"device_map": device_map}

        if load_8bit:
            kwargs['load_in_8bit'] = True
        elif load_4bit:
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        else:
            kwargs['torch_dtype'] = torch.float16

        if 'llava' in model_name.lower():
            # Load LLaVA model
            if 'lora' in model_name.lower() and model_base is not None:
                lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print('Loading LLaVA from base model...')
                model = LlavaLlamaForCausalLM_joint_2st_it_only_ref_instr.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
                if model.lm_head.weight.shape[0] != token_num:
                    model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                    model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

                print('Loading additional LLaVA weights...')
                if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                    non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                else:
                    # this is probably from HF Hub
                    from huggingface_hub import hf_hub_download
                    def load_from_hf(repo_id, filename, subfolder=None):
                        cache_file = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            subfolder=subfolder)
                        return torch.load(cache_file, map_location='cpu')
                    non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
                non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables):
                    non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                model.load_state_dict(non_lora_trainables, strict=False)

                from peft import PeftModel
                print('Loading LoRA weights...')
                model = PeftModel.from_pretrained(model, model_path)
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
                print('Model is loaded...')
            elif model_base is not None:
                # this may be mm projector only
                print('Loading LLaVA from base model...')
                if 'mpt' in model_name.lower():
                    if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                        shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    model = LlavaLlamaForCausalLM_joint_2st_it_only_ref_instr.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                    cfg_pretrained = AutoConfig.from_pretrained(model_path)
                    model = LlavaLlamaForCausalLM_joint_2st_it_only_ref_instr.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            else:
                if 'mpt' in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = LlavaLlamaForCausalLM_joint_2st_it_only_ref_instr.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = LlavaLlamaForCausalLM_joint_2st_it_only_ref_instr.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        else:
            # Load language model
            if model_base is not None:
                # PEFT model
                from peft import PeftModel
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
                print(f"Loading LoRA weights from {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
                print(f"Merging weights")
                model = model.merge_and_unload()
                print('Convert to FP16...')
                model.to(torch.float16)
            else:
                use_fast = False
                if 'mpt' in model_name.lower():
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

        image_processor = None

        if 'llava' in model_name.lower():
            mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
            if mm_use_im_patch_token:
                tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device='cuda', dtype=torch.float16)
            image_processor = vision_tower.image_processor

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        return tokenizer, model, image_processor, context_len
    
    def construct_vision_model(self, path_vision_model_cfg):
        from detectron2.config import LazyConfig
        from llava.model.openseed import build_model
        from llava.model.openseed.BaseModel import BaseModel

        def get_config_from_name(cfg, dataset_name="flickr"):
            # adjust config according to dataset, flickr by default
            if 'sam' in dataset_name:
                cfg.update(cfg['SAM'])
                return cfg
            elif 'flickr' in dataset_name:
                cfg.update(cfg['flickr'])
                return cfg
            elif 'coco_instruct_train' in dataset_name:
                cfg.update(cfg['coco_instruct'])
                return cfg
            elif 'lisa' in dataset_name:
                cfg.update(cfg['LISA_REF'])
                return cfg
            elif 'llava' in dataset_name:
                cfg.update(cfg['llava'])
                return cfg
            elif 'vg' in dataset_name:
                cfg.update(cfg['vg'])
                return cfg
            elif 'part' in dataset_name and 'pascal_part' not in dataset_name and 'partimagenet' not in dataset_name:
                cfg.update(cfg['part'])
                return cfg
            elif 'pascal' in dataset_name or 'paco' in dataset_name or 'partimagenet' in dataset_name :
                cfg.update(cfg['PSACAL_PART'])
                return cfg
            elif 'coco' in dataset_name and 'refonly' in dataset_name:
                # if 'COCO' in cfg.keys():
                cfg.update(cfg['COCO_REF'])
                return cfg
            elif 'refcoco' in dataset_name or "flickr_val" in dataset_name:
                cfg.update(cfg['REF'])
                return cfg
            elif 'coco' in dataset_name:
                if 'COCO' in cfg.keys():
                    cfg.update(cfg['COCO'])
                return cfg
            elif "mapillary" in dataset_name:
                if 'MAPILLARY' in cfg.keys():
                    cfg.update(cfg['MAPILLARY'])
                return cfg
            elif 'ade' in dataset_name:
                if 'ADE20K' in cfg.keys():
                    cfg.update(cfg['ADE20K'])
                return cfg
            elif 'imagenet' in dataset_name:
                if 'IMAGENET' in cfg.keys():
                    cfg.update(cfg['IMAGENET'])
                return cfg
            elif 'vlp' in dataset_name:
                cfg.update(cfg['VLP'])
                return cfg
            elif 'sun' in dataset_name:
                cfg.update(cfg['SUN'])
                return cfg
            elif 'object365' in dataset_name:
                cfg.update(cfg['OBJECT365'])
                return cfg
            elif 'scan' in dataset_name:
                cfg.update(cfg['SCAN'])
                return cfg
            elif 'cityscape' in dataset_name:
                cfg.update(cfg['CITY'])
                return cfg
            elif 'bdd' in dataset_name:
                cfg.update(cfg['BDD'])
                return cfg
            else:
                assert False, "dataset not support."
        self.cfg_vision_model = LazyConfig.load(path_vision_model_cfg)
        vision_model = BaseModel(self.cfg_vision_model, build_model(self.cfg_vision_model))
        vision_model.eval()
        self.model.seg_model = vision_model
        self.model.seg_model.model = self.model.seg_model.model.to(self.model.device)

        self.cfg_inter_model = LazyConfig.load(self.path_inter_model_cfg)
        self.model.initialize_interactive_modules(self.cfg_inter_model)
        self.model.interactive_model.model = self.model.interactive_model.model.to(self.model.device)
        # print("Configuring for Dataset Mapper ...")
        self.cfg_vision_model = get_config_from_name(self.cfg_vision_model)
    @torch.inference_mode()
    def evaluate_sample(self, input_data):
        text, boxes, masks, mask_inter = self.model.forward_eval(input_data)

        return text, boxes, masks, mask_inter
    
def formatting(text, boxes, question_id):
    def find_start_idxes(sentence, word):
        window_size = len(word)
        start_indexes = []
        assert len(sentence) > window_size
        if sentence == window_size:
            return [0]
        for start_index in range(len(sentence) - window_size+1):
            if sentence[start_index: start_index + window_size] == word:
                start_indexes.append(start_index)
        return start_indexes
    def extract_text(sentence):
        # Use regular expression to find and extract the text and number
        import re
        pattern = r"<g_s>|<g_e>"
        cleaned_text = re.sub(pattern, '', sentence)
        return cleaned_text
    def multiboxes_to_str(boxes):
        boxes_text = []
        for box in boxes:
            boxes_text.append(list_to_str(box))
        output_string = ";".join(boxes_text)
        return output_string.replace("];[", ";")
    def list_to_str(list_):
        list_str = [str(round(aa, 3)) for aa in list_]
        return "[" + ",".join(list_str) + "]"
    def format_sentence(splitted_sentence):
        joint_sentence = " ".join(splitted_sentence)
        return joint_sentence
    
    text_pure = ""
    text_boxes = ""
    boxes_pure = []
    
    number = 0
    seg_start_index = find_start_idxes(text, "<seg>")
    if len(seg_start_index) > 0:
        # text = text[:tail_start_index[0]]
        subtexts = text.split(" <seg>")
        for subtext in subtexts:
            if "<g_s>" in subtext:
                # subtext += "<g_e>"
                start_idx = find_start_idxes(subtext, "<g_s>")[0]
                text_pure = format_sentence([text_pure, format_sentence(subtext[:start_idx].split())])
                text_boxes = format_sentence([text_boxes, format_sentence(subtext[:start_idx].split())])
                text_ = extract_text(subtext[start_idx:])
                text_pure = format_sentence([text_pure, format_sentence(text_.split())])
                if number >= len(boxes):
                    print("Error, There should be a wrong prediction.")
                    text_boxes = format_sentence([text_boxes, format_sentence(text_.split())])
                    number += 1
                    continue
                text_boxes = format_sentence([text_boxes, format_sentence(text_.split()) + multiboxes_to_str(boxes[number].cpu().tolist())])
                boxes_pure.append(multiboxes_to_str(boxes[number].cpu().tolist()))
                number += 1
            else:
                text_pure = format_sentence([text_pure, format_sentence(subtext.split())])
                text_boxes = format_sentence([text_boxes, format_sentence(subtext.split())])
        return {
            "question_id": question_id,
            "text": text_pure, 
            "text_boxes": text_boxes, 
            "boxes": boxes_pure,
        }
    else:
        return {
            "question_id": question_id,
            "text": text, 
            "text_boxes": text, 
            "boxes": []
        }

def evaluate_(path_benchmarks, dir_image, evaluator, matching_threshold):
    def unresize_box(box, width, height, size):
        # ori_size = max(width, height)
        # ratio = ori_size / size
        ratio = min(width, height) / max(width, height)
        if width > height:  # then the height dimension is padded, the y coordinates should be divided by ratio
            box[:, 1] = box[:, 1] / ratio
            box[:, 3] = box[:, 3] / ratio
        elif width < height:  # then the height dimension is padded, the y coordinates should be divided by ratio
            box[:, 0] = box[:, 0] / ratio
            box[:, 2] = box[:, 2] / ratio
        return box
    def filter_empty_box(text, boxes_image):
        def extract_text(sentence):
            # Use regular expression to find and extract the text and number
            import re
            if " <seg>" in sentence:
                pattern = r"<g_s>|<g_e> <seg>"
                cleaned_text = re.sub(pattern, '', sentence)
                return cleaned_text
            else:
                cleaned_text = re.sub(r'<g_s> \d+', '', sentence)
                cleaned_text = re.sub(r' <g_e>', '', cleaned_text)
                return cleaned_text
        
        has_gd = True if "<seg>" in text else False
        if len(boxes_image) == 0:
            return text, boxes_image
        else:
            if has_gd:
                sub_texts = text.split(" <seg>")
                sub_texts_filtered = []
                boxes_image_filtered = []
                for box_per_gd, text_per_gd in zip(boxes_image, sub_texts):
                    text_per_gd += " <seg>"
                    ind_nonempty_box = torch.where(box_per_gd.abs().sum(dim=1)>0)
                    if len(ind_nonempty_box[0]) < box_per_gd.shape[0]:  # empty box encountered
                        if len(ind_nonempty_box[0]) == 0:
                            text_per_gd = " " + " ".join(extract_text(text_per_gd).split())
                            sub_texts_filtered.append(text_per_gd)  # box is desperated
                            continue
                        else:
                            box_per_gd = box_per_gd[ind_nonempty_box]
                            boxes_image_filtered.append(box_per_gd)
                            sub_texts_filtered.append(text_per_gd)
                    else:
                        boxes_image_filtered.append(box_per_gd)
                        sub_texts_filtered.append(text_per_gd)
                sub_texts_filtered.append(sub_texts[-1])
                text_filtered = "".join(sub_texts_filtered)
                return text_filtered, boxes_image_filtered
            else:
                text_filtered = " ".join(extract_text(text).split())
                boxes_image_filtered = []
                
                return text_filtered, boxes_image_filtered
    def debug(image, boxes, prefix):
        import cv2
        # image = cv2.imread(path_image)
        def transform_str2numpy(boxes_str):
            boxes_str = boxes_str.replace(";", "];[")
            boxes_list = []
            for box_str in boxes_str.split(";"):
                box_list = [float(aa) for aa in box_str[1:-1].split(",")]
                boxes_list.append(box_list)
            return boxes_list
        boxes_ = []
        for box in boxes: boxes_.extend(transform_str2numpy(box))
        boxes = torch.tensor(boxes_)
        image = image[..., ::-1]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        height,width,_ = image.shape
        for box in boxes:
            box = (box.cpu() * torch.tensor([width, height, width, height])).int().squeeze()
            box = box.tolist()
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
        cv2.imwrite(f"{prefix}_debug.jpg", image)
    
    datas = load_benchmark(dir_image, path_benchmarks)[:20] #! use first 20 samples for debug.
    data_mapper = evaluator.data_mapper
    device = evaluator.model.device
    outputs = []
    for data in tqdm(datas):
        input_data = data_mapper(data)[0]
        for key, value in input_data.items():
            if isinstance(value, torch.Tensor):
                input_data[key] = value.to(device)
        input_data["matching_threshold"] = matching_threshold
        text, boxes = evaluator.evaluate_sample([input_data])
        text, boxes = filter_empty_box(text, boxes)
        boxes = [unresize_box(bb.detach().cpu(), input_data["width"], input_data["height"], 1024) for bb in boxes]
        output = formatting(text, boxes, input_data["question_id"])
        # from ipdb import set_trace; set_trace()
        # debug(cv2.imread(input_data["file_name"]), output["boxes"], prefix=str(input_data["question_id"]))
        outputs.append(output)
    return outputs

def evaluate(args=None):
    evaluator = Evaluator_MM(
        model_path=args.model_path,
        path_vision_model_cfg=args.vision_model_cfg,
    )
    results = evaluate_(args.path_benchmark, dir_image=args.image_root, evaluator=evaluator, matching_threshold=args.matching_threshold)
    return results
    
if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--model_path", type=str, default="xx")
    args.add_argument("--vision_model_cfg", type=str, default="xx")
    args.add_argument("--matching_threshold", type=float, default=0.2)
    args.add_argument("--path_benchmark", default="./dataset/qa1000_questions.jsonl")
    args.add_argument("--image_root", default="./dataset/coco/val2014")
    args = args.parse_args()
    results = evaluate(args)
    path_save = f"./LLaVA_G_{args.path_benchmark.split('/')[-1].split('.')[0]}_t{args.matching_threshold}.jsonl"  #! sync
    print("Writing at: ", path_save)
    save_jsonl_file(results, path_save)
    # CUDA_VISIBLE_DEVICES=0 python llava/eval/LLaVA_G_Eval.py --model_path ./checkpoints/llava_stage2_new_joint_seg0.1_data_v3/checkpoint-8000 --vision_model_cfg configs/openseed/openseed_swint_lang_coco_instruct_coco_end_llava_bench.yaml --path_benchmark ./dataset/qa90_questions.jsonl --image_root ./dataset/coco/val2014 --matching_threshold 0.2