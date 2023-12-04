# Copyright (c) Facebook, Inc. and its affiliates.
import os
import itertools
import logging
import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.utils.data
import torch.utils.data as torchdata

import detectron2.utils.comm as comm
from detectron2.data.build import (
    build_batch_data_loader,
    load_proposals_into_dataset,
    trivial_batch_collator,
)
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import InferenceSampler, TrainingSampler

from fvcore.common.config import CfgNode
from omegaconf import DictConfig, OmegaConf

from .dataset_mappers import (
    COCOPanopticInteractiveDatasetMapper,
    FlickrNewBaselineDatasetMapper,
    VGNewBaselineDatasetMapper,
    COCOInstructGroundingDatasetMapper,
    COCOInterGroundingDatasetMapper,
)

from .custom_dataset_dataloader import build_custom_test_loader
from llava.model.openseed.utils import configurable
from detectron2.utils.comm import get_world_size, is_main_process
from typing import Any, Dict, List, Set

class JointLoader(torchdata.IterableDataset):
    def __init__(self, loaders, key_dataset):
        dataset_names = []
        for key, loader in loaders.items():
            name = "{}".format(key.split('_')[0])
            # name = "{}".format(key)
            setattr(self, name, loader)
            dataset_names += [name]
        self.dataset_names = dataset_names
        self.key_dataset = key_dataset
    
    def __iter__(self):
        for batch in zip(*[getattr(self, name) for name in self.dataset_names]):
            yield {key: batch[i] for i, key in enumerate(self.dataset_names)}

    def __len__(self):
        return len(getattr(self, self.key_dataset))

def filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if isinstance(ann, list):
                for instance in ann:
                    if instance.get("iscrowd", 0) == 0:
                        return True
            else:
                if ann.get("iscrowd", 0) == 0:
                    return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def get_detection_dataset_dicts(
    dataset_names, filter_empty=True, proposal_files=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts, dataset_names)

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=None,
    )
    # import ipdb;ipdb.set_trace()
    if mapper is None:
        if isinstance(cfg, (DictConfig)):
            cfg = OmegaConf.to_container(copy.deepcopy(cfg))
        mapper_cfg = CfgNode({'INPUT': cfg['INPUT'], 'MODEL': cfg['MODEL'], 'DATASETS': cfg['DATASETS']})
        mapper = DatasetMapper(mapper_cfg, False)
    assert cfg['TEST']['BATCH_SIZE_TOTAL'] % get_world_size() == 0, "Evaluation total batchsize is not divisible by gpu number"
    batch_size = cfg['TEST']['BATCH_SIZE_TOTAL'] // get_world_size()

    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg['DATALOADER']['NUM_WORKERS'],
        "sampler": InferenceSampler(len(dataset)),
        "batch_size": batch_size,
    }


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset: Union[List[Any], torchdata.Dataset],
    *,
    mapper: Callable[[Dict[str, Any]], Any],
    sampler: Optional[torchdata.Sampler] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> torchdata.DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper: a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler: a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )


def _train_loader_from_config(cfg, dataset_name, mapper, *, dataset=None, sampler=None):
    cfg_datasets = cfg['DATASETS']
    cfg_dataloader = cfg['DATALOADER']
    
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg_dataloader['FILTER_EMPTY_ANNOTATIONS'],
            proposal_files=cfg_datasets['PROPOSAL_FILES_TRAIN'] if cfg_dataloader['LOAD_PROPOSALS'] else None,
        )

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg_dataloader['SAMPLER_TRAIN']
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        sampler = TrainingSampler(len(dataset))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg['TRAIN']['BATCH_SIZE_TOTAL'],
        "aspect_ratio_grouping": cfg_dataloader['ASPECT_RATIO_GROUPING'],
        "num_workers": cfg_dataloader['NUM_WORKERS'],
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size, aspect_ratio_grouping=True, num_workers=0
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


def get_config_from_name(cfg, dataset_name):
    # adjust config according to dataset
    if 'sam' in dataset_name:
        cfg.update(cfg['SAM'])
        return cfg
    elif 'flickr' in dataset_name:
        cfg.update(cfg['flickr'])
        return cfg
    elif 'coco_instruct' in dataset_name:
        cfg.update(cfg['coco_instruct'])
        return cfg
    elif 'coco_interactive' in dataset_name:
        cfg.update(cfg['coco_interactive'])
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



def build_train_dataloader(cfg,tokenizer=None,data_args=None,preprocess=None,llava_cap_loader=None ):
    dataset_names = cfg['DATASETS']['TRAIN']
    
    loaders = {}
    cfg = copy.deepcopy(cfg)
    for dataset_name in dataset_names:
        cfg = get_config_from_name(cfg, dataset_name)
        mapper_name = cfg['INPUT']['DATASET_MAPPER_NAME']

        if mapper_name =="flickr":
            mapper=FlickrNewBaselineDatasetMapper(cfg,True,tokenizer=tokenizer,data_args=data_args,preprocess=preprocess)
            loaders['flickr'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name =="coco_instruct":
            mapper=COCOInstructGroundingDatasetMapper(cfg,True,tokenizer=tokenizer,data_args=data_args,preprocess=preprocess)
            loaders['coco_instruct'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name =="coco_interactive":
            if "refcoco" in dataset_name:
                refcoco=True
            else:
                refcoco=False
            mapper=COCOInterGroundingDatasetMapper(cfg,True,tokenizer=tokenizer,data_args=data_args,preprocess=preprocess,refcoco=refcoco)
            if refcoco:
                loaders['interactiveref'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
            else:
                loaders['interactive'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name =="vg":
            mapper=VGNewBaselineDatasetMapper(cfg,True,tokenizer=tokenizer,data_args=data_args,preprocess=preprocess)
            loaders['vg'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        elif mapper_name == "coco_ref_panoptic_lsj":
            mapper = COCOPanopticInteractiveDatasetMapper(cfg, cfg.get('Train',True),tokenizer=tokenizer,data_args=data_args,preprocess=preprocess)
            loaders['refcoco'] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)
        else:
            mapper = None
            loaders[dataset_name] = build_detection_train_loader(cfg, dataset_name=dataset_name, mapper=mapper)

    if llava_cap_loader is not None:
        loaders['llava_cap'] = llava_cap_loader
    if len(loaders) == 1 and not cfg['LOADER'].get('JOINT', False):
        for k, v in loaders.items():
            print("number of iterations per epoch: ", v, len(loaders[k]))
        return list(loaders.values())[0]
    else:
        return JointLoader(loaders, key_dataset=cfg['LOADER'].get('KEY_DATASET', 'coco'))