# Copyright (c) Facebook, Inc. and its affiliates.
# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
import json
import os
import collections

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager
import pycocotools.mask as mask_util


_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {


    "coco_instruct_train_v3": (
        "coco/train2014", # image_root
        "coco/annotations/instances_train2017_gvc.json", # annot_root
        "llava/annotations/grounded_visual_chat_data.json",
    ),

    "coco_interactive": (
        "coco/train2014", # image_root
        "coco/annotations/instances_train2014_filter.json", # annot_root
        "llava/annotations/llava_instruct_150k_visual_prompt.json",
    ),
    "coco_interactive_refcoco": (
        "coco/train2017", # image_root
        "coco/annotations/instances_train2017_refcoco.json", # annot_root
        "coco/annotations/grounding_train2017_instruct.json",
    ),
}


def get_metadata():
    meta = {}
    return meta


def load_coco_json(image_root, annot_json,conversation, metadata):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    with PathManager.open(annot_json) as f:
        json_info = json.load(f)
        
    # build dictionary for grounding
    grd_dict = collections.defaultdict(list)

    imgid2image = {}
    for image in json_info["images"]:
        image_id = image["id"]
        imgid2image[image_id] = image
    for grd_ann in json_info['annotations']:
        image_id = int(grd_ann["image_id"])
        segm = grd_ann.get("segmentation", None)
        if segm:  # either list[list[float]] or dict(RLE)
            if isinstance(segm, dict):
                if isinstance(segm["counts"], list):
                    # convert to compressed RLE
                    segm = mask_util.frPyObjects(segm, *segm["size"])

            grd_ann["segmentation"] = segm
        grd_dict[image_id].append(grd_ann)

    conv_dict = collections.defaultdict(list)
    with open(conversation) as f:
        data = json.load(f)
    for d in data:
        image_id = int(d['id'])
        if 'gd_ls' not in d:
            d['gd_ls']=None
        if 'q_gd_ls' in d:
            conv_dict[image_id].append((d['conversations'],d['q_gd_ls']))
        else:
            conv_dict[image_id].append((d['conversations'], d['gd_ls']))

    ret = []
    for d in data:
        image_id = int(d['id'])
        image= imgid2image[image_id]
        image_file = os.path.join(image_root, image['file_name'])
        grounding_anno = grd_dict[image_id]
        if image_id in conv_dict and len(conv_dict[image_id])>0:
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "grounding_info": grounding_anno,
                    "conversations": conv_dict[image_id],
                }
            )

    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_coco(
    name, metadata, image_root, annot_json,conversation):
    DatasetCatalog.register(
        name,
        lambda: load_coco_json(image_root, annot_json,conversation, metadata),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_coco(root):
    for (
        prefix,
        (image_root, annot_root,conversation_path),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        register_coco(
            prefix,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, annot_root),
            conversation_path,
        )

_root = os.getenv("DATASET", "datasets")
register_all_coco(_root)
