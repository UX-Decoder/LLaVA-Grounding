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
from detectron2.utils.file_io import PathManager


_PREDEFINED_SPLITS = {

    "flickr_val": (
        "goldg/flickr30k_entities/val", # image_root
        "final_flickr_separateGT_val.json", # # anno_path
    ),
    "flickr_train": (
        "goldg/flickr30k_entities/train", # image_root
        "final_flickr_separateGT_train.json", # # anno_path
    ),
}


def get_metadata():
    meta = {}
    return meta


def load_flickr_json(image_root, annot_json, metadata):


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
        caption=image['caption']
        image_file = os.path.join(image_root, image['file_name'])
        grounding_anno = grd_dict[image_id]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "grounding_info": grounding_anno,
                "caption": caption,
            }
        )
    assert len(ret), f"No images found in {image_root}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    return ret


def register_flickr(
    name, metadata, image_root, annot_json):
    DatasetCatalog.register(
        name,
        lambda: load_flickr_json(image_root, annot_json, metadata),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        json_file=annot_json,
        evaluator_type="grounding_refcoco",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


def register_all_flickr(root,anno_root):
    for (
        prefix,
        (image_root, anno_path),
    ) in _PREDEFINED_SPLITS.items():
        register_flickr(
            prefix,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root,anno_root, anno_path),
        )

_root = os.getenv("DATASET", "datasets")
ann_root = os.getenv("Flickr", "flickr30k_entities/annotations")
register_all_flickr(_root,ann_root)
