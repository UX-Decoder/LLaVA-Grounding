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


_PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION = {

    "vg_train": (
        "/comp_robot/zhanghao/datasets/vg/images/", # image_root
        "/comp_robot/zhanghao/datasets/vg/annotations/train.json", # annot_root
    ),
}


def get_metadata():
    meta = {}
    return meta


def load_refcoco_json(image_root, annot_json, metadata):
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
    # for grd_ann in json_info['annotations']:
    #     image_id = int(grd_ann["image_id"])
    #     grd_dict[image_id].append(grd_ann)
    for grd_ann in json_info['annotations']:
        image_id = int(grd_ann["image_id"])
        grd_dict[image_id].append(grd_ann)

    ret = []
    for image in json_info["images"]:
        image_id = int(image["id"])
        # caption=image['caption']
        image_file = os.path.join(image_root, image['file_name'])
        grounding_anno = grd_dict[image_id]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "annotations": grounding_anno,
            }
        )
    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_refcoco(
    name, metadata, image_root, annot_json):
    DatasetCatalog.register(
        name,
        lambda: load_refcoco_json(image_root, annot_json, metadata),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_refcoco(root,ann_root):
    for (
        prefix,
        (image_root, annot_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC_CAPTION.items():
        register_refcoco(
            prefix,
            get_metadata(),
            image_root if image_root.startswith("/") else os.path.join(root, image_root),
            annot_root if annot_root.startswith("/") else os.path.join(ann_root, annot_root),
        )

_root = os.getenv("DATASET", "datasets")
ann_root = os.getenv("Flickr", "/comp_robot/zhanghao/goldg/flickr30k_entities/annotations")
register_all_refcoco(_root,ann_root)
