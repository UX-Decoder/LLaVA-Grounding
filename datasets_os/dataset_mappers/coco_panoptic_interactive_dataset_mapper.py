# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import random

import numpy as np
import torch
import PIL.Image as Image
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode
from detectron2.structures.boxes import pairwise_iou
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import MetadataCatalog
from pycocotools import mask as coco_mask
from utils.prompt_engineering import prompt_engineering, get_prompt_templates
from llava.model.openseed.utils import configurable
# from ..shapes.sampler import build_shape_sampler

__all__ = ["COCOPanopticInteractiveDatasetMapper"]

def filter_empty_instances_by_box(
        instances, by_box=True, by_mask=False, box_threshold=1e-5, return_mask=False
):
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    if return_mask:
        return instances[m], m
    return instances[m]

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    # assert is_train, "Only support training augmentation"
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


# This is specifically designed for the COCO dataset.
class COCOPanopticInteractiveDatasetMapper:
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
            caption_thres,
            # lvis,
            # lvis_thres,
            max_grounding_num,
            tokenizer,
            data_args,
            preprocess,
            # shape_sampler,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[COCOPanopticNewBaselineDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.caption_thres = caption_thres
        self.grounding = True
        # self.lvis = lvis
        # self.lvis_thres = lvis_thres
        self.max_grounding_num = max_grounding_num
        self.caption_similarity = torch.load(MetadataCatalog.get('logistic').get('caption_similarity_pth'))
        self.tokenizer = tokenizer
        self.processor = data_args.image_processor
        self.data_args = data_args
        self.preprocess = preprocess

        # self.shape_sampler = shape_sampler

    @classmethod
    def from_config(cls, cfg, is_train=True,tokenizer=None,data_args=None,preprocess=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        # shape_sampler = build_shape_sampler(cfg)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "caption_thres": cfg['MODEL']['DECODER']['CAPTION']['SIM_THRES'],
            "max_grounding_num": cfg['MODEL']['DECODER']['GROUNDING']['MAX_LEN'],
            "tokenizer": tokenizer,
            "data_args": data_args,
            "preprocess": preprocess,
            # "shape_sampler": shape_sampler,
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
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image_ori"]=image
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))


        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])

            # is_things = [COCO_CATEGORIES[idx]['isthing'] for idx in classes]
            classes = np.array(classes)
            # is_things = np.array(is_things)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            # instances.is_things = torch.tensor(is_things, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                masks = BitMasks(torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])))
                instances.gt_masks = masks
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks
                instances.gt_boxes = masks.get_bounding_boxes()

        if self.grounding:
            grounding_anno = dataset_dict['grounding_info']
            if self.is_train:
                grounding_len = random.randint(1, self.max_grounding_num - 1)
            else:
                grounding_len = 1
            if len(grounding_anno) > 0:
                masks_grd = []
                texts_grd = []
                mode = 'text'
                random.shuffle(grounding_anno)
                for ann in grounding_anno:
                    rle = coco_mask.frPyObjects(
                        ann['segmentation'], dataset_dict['height'], dataset_dict['width'])
                    m = coco_mask.decode(rle)
                    # sometimes there are multiple binary map (corresponding to multiple segs)
                    m = np.sum(m, axis=2)>0
                    m = m.astype(np.uint8)  # convert to np.uint8
                    m = transforms.apply_segmentation(m[:, :, None])[:, :, 0]==1
                    masks_grd += [m]
                    # random select a sentence of a single annotation.
                    rand_index = random.randint(0, len(ann['sentences']) - 1)
                    texts_grd += [ann['sentences'][rand_index]['raw'].lower()]
                max_len = min(grounding_len, len(texts_grd))
                indices = np.random.permutation(max_len)
                texts_grd = list(np.array(texts_grd)[indices])
                masks_grd = torch.tensor(np.stack(masks_grd)[indices])
                hash_grd = np.array([hash(txt) for txt in texts_grd])
                gt_classes = list(range(len(texts_grd)))
                gt_classes = [[lb] for lb in gt_classes]
                label_set=texts_grd
            else:
                assert self.is_train
                masks_grd = instances.gt_masks.tensor
                mode = 'class'
                assert len(masks_grd) > 0

                texts_grd = np.array([COCO_CATEGORIES[idx]['name'] for idx in classes])
                hash_grd = np.array([hash(txt) for txt in texts_grd])
                unique_hash_grd = np.unique(hash_grd)
                np.random.shuffle(unique_hash_grd)
                max_len = min(grounding_len,len(unique_hash_grd))
                indices = np.random.permutation(max_len)
                selected_unique_hash_grd = unique_hash_grd[indices]
                selected_mask = np.in1d(hash_grd, selected_unique_hash_grd)
                texts_grd = texts_grd[selected_mask]
                hash_grd = hash_grd[selected_mask]
                masks_grd = masks_grd[selected_mask]
                texts_grd = [
                    text.replace('-other', '').replace('-merged', '').replace('-stuff', '')
                    for text in texts_grd]
                label_set=list(set(texts_grd))
                gt_classes=[[label_set.index(lb)] for lb in texts_grd]

            instances_gd = Instances(image_shape)
            instances_gd.gt_masks = BitMasks(masks_grd)
            instances_gd.gt_boxes = BitMasks(masks_grd).get_bounding_boxes()
            instances_gd.gt_masks=instances_gd.gt_masks.tensor
            instances_gd.gt_classes=gt_classes
            dataset_dict["instances"] = instances_gd
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