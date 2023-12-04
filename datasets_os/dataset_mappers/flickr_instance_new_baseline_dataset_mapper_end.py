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
        dataset_dict["image_ori"]=image
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)

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

        if "grounding_info" in dataset_dict:

            for obj in dataset_dict['grounding_info']:
                obj["bbox_mode"] = BoxMode.XYWH_ABS
                obj['tokens']=dataset_dict['caption'][obj['tokens_positive'][0][0]:obj['tokens_positive'][0][1]]
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict["grounding_info"]
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            assert len(annos)>0
            assert  "segmentation" in annos[0]
            instances = utils.annotations_to_instances(annos, image_shape,mask_format="bitmask")
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

            span_set = dict()
            end_dict = dict()
            gt_classes= []
            for i, info in enumerate(dataset_dict['grounding_info']):
                gt_classes.append([])
                # if len(info['tokens_positive'])>1:
                #     print("multi class")
                for j in range(len(info['tokens_positive'])):
                    if info['tokens_positive'][j][0] in span_set:
                        span_set[info['tokens_positive'][j][0]].append(i)
                    else:
                        span_set[info['tokens_positive'][j][0]] = [i]
                    if info['tokens_positive'][j][0] in end_dict:
                        assert end_dict[info['tokens_positive'][j][0]] == info['tokens_positive'][j][1]
                    else:
                        end_dict[info['tokens_positive'][j][0]] = info['tokens_positive'][j][1]
                    gt_classes[-1].append(info['tokens_positive'][j][0])

            end_dict = sorted(end_dict.items())
            start2id = dict()
            for i, (s, e) in enumerate(end_dict):
                start2id[s] = i
            gt_classes= [[start2id[s] for s in gt_class] for gt_class in gt_classes]
            instances.gt_classes = gt_classes
            dataset_dict["instances"] = instances
            # span_list = sorted(span_set.items())

            # for k, v in span_set:
            #     for i in range(len(v)):
            #         v[i] = positive_new_ids[v[i]]
            cap_pieces = []
            last_e = 0
            for s, e in end_dict:
                cap_pieces.append(dataset_dict['caption'][last_e:s])
                cap_pieces.append(dataset_dict['caption'][s:e])
                last_e = e
            cap_pieces.append(dataset_dict['caption'][last_e:])
            new_cap = []
            for i, piece in enumerate(cap_pieces):
                if i % 2 == 1:
                    piece = '<g_s>' + piece + '<g_e>'
                new_cap.append(piece)
            new_cap = "".join(new_cap)
            tail = [f'{i + 1}: <seg>' for i in range(new_cap.count("<g_s>"))]
            tail = '; '.join(tail)
            new_cap += f' {tail}.'
            # gt_ids = []
            # for s, e in end_dict:
            #     if len(span_set[s]) > 1:
            #         return dataset_dict
            #     gt_ids.append(span_set[s][0] + 1)
            # ground_annos = dict()
            # ground_annos['gt_ids'] = gt_ids
            # ground_annos['gt_anno_ids'] = [dataset_dict['grounding_info'][gt_id_ - 1]['id'] for gt_id_ in gt_ids]
            # ground_annos['caption'] = new_cap
            question={'from': 'human', 'value': "<image>\nPresent a compact description of the photo's key features.\n(with grounding)"}
            answer={'from': 'gpt', 'value': new_cap}
            dataset_dict['conversation'] = [[question, answer]]
            # sources = preprocess_multimodal(
            #     copy.deepcopy(dataset_dict['conversation']),
            #     self.data_args)
            data_dict_conversation = self.preprocess(
                dataset_dict['conversation'],
                self.tokenizer,
                has_image=True)
            data_dict_conversation = dict(input_ids=data_dict_conversation["input_ids"][0],
                             labels=data_dict_conversation["labels"][0])
            dataset_dict.update(data_dict_conversation)
            dataset_dict['tokenizer']=self.tokenizer

        return dataset_dict
