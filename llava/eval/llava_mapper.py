# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch
import PIL.Image as Image
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances

from pycocotools import mask as coco_mask

from llava.model.openseed.utils import configurable
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# from llava.train.train_hao_seg_flickr import ,preprocess
__all__ = ["COCOInstanceNewBaselineDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def preprocess_multimodal(
    sources,
    is_multimodal
):
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if False:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
        cfg_input = cfg['INPUT']
        image_size = cfg_input['IMAGE_SIZE']
        min_scale = cfg_input['MIN_SCALE']
        max_scale = cfg_input['MAX_SCALE']

        augmentation = []

        # if cfg_input['RANDOM_FLIP'] != "none":
        #     augmentation.append(
        #         T.RandomFlip(
        #             horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
        #             vertical=cfg_input['RANDOM_FLIP'] == "vertical",
        #         )
        #     )

        augmentation.extend([
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ])
    else:
        cfg_input = cfg['INPUT']
        image_size = cfg_input['IMAGE_SIZE']
        min_scale = cfg_input['MIN_SCALE']
        max_scale = cfg_input['MAX_SCALE']

        augmentation = []

        # if cfg_input['RANDOM_FLIP'] != "none":
        #     augmentation.append(
        #         T.RandomFlip(
        #             horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
        #             vertical=cfg_input['RANDOM_FLIP'] == "vertical",
        #         )
        #     )

        augmentation.extend([
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ])

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOInstanceNewBaselineDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        tokenizer,
        image_processor,
        preprocess,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOInstanceNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.processor = image_processor
        self.preprocess = preprocess
    
    @classmethod
    def from_config(cls, cfg, is_train=True,tokenizer=None,image_processor=None,preprocess=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "tokenizer": tokenizer,
            "image_processor": image_processor,
            "preprocess": preprocess,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        #########llava image processing

        image_clip = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        dataset_dict["image_clip"] = image_clip

        ##################

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        dataset_dict["image_ori"]=image
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        num_conversations = len(dataset_dict['conversations'])
        rd = np.random.choice(num_conversations)
        # selected_conversation, grounding_list = dataset_dict['conversations'][rd]
        # dataset_dict['conversation'] = [selected_conversation]
        selected_conversation = [aa[0] for aa in dataset_dict['conversations']]
        dataset_dict['conversation'] = selected_conversation
        sources = preprocess_multimodal(
            copy.deepcopy(dataset_dict['conversation']),
            True)  #! Debug here
        # sources = copy.deepcopy(dataset_dict['conversation'])
        data_dict_conversation = self.preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        data_dict_conversation = dict(input_ids=data_dict_conversation["input_ids"][0],
                                      labels=data_dict_conversation["labels"][0])
        dataset_dict.update(data_dict_conversation)
        dataset_dict['tokenizer'] = self.tokenizer
        num_segs = 1 # sum([conv['value'].count('<seg>') for conv in selected_conversation])
        # grounding_list=
        if "grounding_info" in dataset_dict and len(dataset_dict['grounding_info'])>0:
            anno_id2id=dict()
            for id,obj in enumerate(dataset_dict['grounding_info']):
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                anno_id2id[obj['id']]=id
            id2class=[[] for _ in range(len(dataset_dict['grounding_info']))]

            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict["grounding_info"]
            ]
            # assert  "segmentation" in annos[0]
            instances = utils.annotations_to_instances(annos, image_shape,mask_format="bitmask")

            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                # gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks.tensor

            if grounding_list is None:
                dataset_dict['grounding']=False
                grounding_mask=[False for _ in range(num_segs)]
                dataset_dict['grounding_mask']=grounding_mask
            else:
                grounding_mask=[True if g is not None else False for g in grounding_list]
                dataset_dict['grounding_mask']=grounding_mask
                new_grounding_list=[g for g in grounding_list if g is not None]
                if sum(grounding_mask)==0:
                    dataset_dict['grounding']=False
                else:
                    dataset_dict['grounding']=True
            if dataset_dict['grounding']:
                # assert num_segs == len(grounding_list)
                for grounding_id,grounding in enumerate(new_grounding_list):
                    if grounding is not None:
                        for annid in grounding:
                            id2class[anno_id2id[annid]].append(grounding_id)

                instances.gt_classes=id2class
            dataset_dict["instances"] = instances

        else:
            dataset_dict['grounding'] = False
            grounding_mask = [False for _ in range(num_segs)]
            dataset_dict['grounding_mask'] = grounding_mask

        return [dataset_dict]
