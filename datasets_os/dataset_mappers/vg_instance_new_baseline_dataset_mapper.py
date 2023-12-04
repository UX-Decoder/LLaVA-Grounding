# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances

from pycocotools import mask as coco_mask

from llava.model.openseed.utils import configurable
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes

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
        max_grounding_num,
        tokenizer,
        data_args,
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
        self.max_grounding_num = max_grounding_num
        self.tokenizer = tokenizer
        self.processor = data_args.image_processor
        self.data_args = data_args
        self.preprocess = preprocess
    
    @classmethod
    def from_config(cls, cfg, is_train=True,tokenizer=None,data_args=None,preprocess=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "max_grounding_num": cfg['MODEL']['DECODER']['GROUNDING']['MAX_LEN'],
            "tokenizer": tokenizer,
            "data_args": data_args,
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

        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image_clip = expand2square(image, tuple(int(x * 255) for x in self.processor.image_mean))
            image_clip = self.processor.preprocess(image_clip, return_tensors='pt')['pixel_values'][0]
        else:
            image_clip = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        dataset_dict["image_clip"] = image_clip

        ##################
        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        dataset_dict["image_ori"]=image
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        assert "annotations" in dataset_dict

        for obj in dataset_dict['annotations']:
            obj["bbox_mode"] = BoxMode.XYWH_ABS
        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict["annotations"]
        ]
        # NOTE: does not support BitMask due to augmentation
        # Current BitMask cannot handle empty objects
        assert len(annos)>0
        # assert  "segmentation" in annos[0]
        instances = utils.annotations_to_instances(annos, image_shape,mask_format="bitmask")
        instances.captions=[ann['caption'] for ann in dataset_dict["annotations"]]
        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        # Need to filter empty instances first (due to augmentation)
        # instances = utils.filter_empty_instances(instances)
        # Generate masks from polygon
        h, w = instances.image_size
        # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
        if hasattr(instances, 'gt_masks'):
            gt_masks = instances.gt_masks
            # gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks.tensor
        # dataset_dict["instances"] = instances
        num_instances = len(instances)
        indices = list(range(num_instances))
        import random
        if self.is_train:
            grounding_len = random.randint(1, self.max_grounding_num - 1)
        else:
            grounding_len = 1
        random.shuffle(indices)
        indices = indices[:grounding_len]
        texts_grd = [instances.captions[i] for i in indices]
        gt_classes = list(range(len(texts_grd)))
        gt_classes = [[lb] for lb in gt_classes]
        label_set = texts_grd
        grounding_instances = Instances(image_size=(h, w))
        grounding_instances.gt_boxes = instances.gt_boxes[indices]
        grounding_instances.gt_classes = gt_classes
        dataset_dict["instances"]=grounding_instances
        conversations=[]
        for i in range(len(label_set)):
            if i==0:
                question={'from': 'human', 'value': f"<image>\n Please detect the object according to the text {label_set[i]} (referring)."}
            else:
                question={'from': 'human', 'value': f"Please detect the object according to the text {label_set[i]} (referring)."}
            answer={'from': 'gpt', 'value': '<seg> .'}
            conversations.append(question)
            conversations.append(answer)

        dataset_dict['conversation'] = [conversations]

        data_dict_conversation = self.preprocess(
            dataset_dict['conversation'],
            self.tokenizer,
            has_image=True)
        data_dict_conversation = dict(input_ids=data_dict_conversation["input_ids"][0],
                         labels=data_dict_conversation["labels"][0])
        dataset_dict.update(data_dict_conversation)
        dataset_dict['tokenizer']=self.tokenizer

        return dataset_dict
