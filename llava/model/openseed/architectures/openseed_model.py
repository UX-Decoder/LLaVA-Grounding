# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from MaskDINO https://github.com/IDEA-Research/MaskDINO by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .registry import register_model
from ..utils import configurable, box_ops #, get_class_names
from ..backbone import build_backbone, Backbone
from ..body import build_openseed_head
from ..modules import sem_seg_postprocess, HungarianMatcher, SetCriterion

from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
import random

class OpenSeeD(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        train_dataset_name: str,
        background: bool,
        coco_on=True,
        coco_mask_on=True,
        o365_on=True,
        merge_class=False,
        coco_only=False,
        detach_seg=False,
        eval_train=False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.detach_seg=detach_seg
        self.eval_train=eval_train
        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss

        self.train_class_names = dict()
        self.train_dataset_name = train_dataset_name
        self.coco_mask_on = coco_mask_on
        self.task_switch = {'coco': coco_on, 'o365': o365_on}
        self.num_correct_gd=0
        self.num_total_gd=0
        self.num_correct_ref = 0
        self.num_total_ref = 0
        self.num_correct_coco = 0
        self.num_total_coco = 0
        self.coco_only=coco_only
        self.loss_dict=None
        self.mean_iou=0.0
        ########
        self.total_union=0.0
        self.total_intersection=0.0
        # self.cIoU=0.0
        print("self.task_switch ", self.task_switch)
        # HACK for only two datasets for seg and det
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        # Loss parameters:
        deep_supervision = dec_cfg['DEEP_SUPERVISION']
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        # loss weights
        class_weight = dec_cfg['CLASS_WEIGHT']
        cost_class_weight = dec_cfg['COST_CLASS_WEIGHT']
        cost_dice_weight = dec_cfg['COST_DICE_WEIGHT']
        dice_weight = dec_cfg['DICE_WEIGHT']
        cost_mask_weight = dec_cfg['COST_MASK_WEIGHT']
        mask_weight = dec_cfg['MASK_WEIGHT']
        cost_box_weight = dec_cfg['COST_BOX_WEIGHT']
        box_weight = dec_cfg['BOX_WEIGHT']
        cost_giou_weight = dec_cfg['COST_GIOU_WEIGHT']
        giou_weight = dec_cfg['GIOU_WEIGHT']

        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
        )

        # MaskDINO losses and weight_dict
        weight_dict = {"loss_mask_cls_0": class_weight}
        weight_dict.update({"loss_mask_bce_0": mask_weight, "loss_mask_dice_0": dice_weight})
        weight_dict.update({"loss_bbox_0":box_weight,"loss_giou_0":giou_weight})
        # two stage is the query selection scheme
        if dec_cfg['TWO_STAGE']:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = dec_cfg['DN']
        dn = 'no'
        # TODO hack for dn lable loss
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["dn_labels", "boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["dn_labels", "masks", "boxes"]
        else:
            dn_losses=[]
        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k.replace('_0', '_{}'.format(i+1)): v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if dec_cfg['BOX']:
            losses = ["labels", "masks","boxes"]
        else:
            losses = ["labels", "masks"]

        # update task switch
        task_switch = {}
        task_switch.update({'bbox': dec_cfg.get('DETECTION', True), 'mask': dec_cfg.get('MASK', True)})
        top_x_layers = {'mask': dec_cfg.get('TOP_MASK_LAYERS', 10),
                        'box': dec_cfg.get('TOP_DETECTION_LAYERS', 10)}
        weight_multiplier= dec_cfg.get('WEIGHT_MULTIPLIER', 1.0)
        weight_dict={k:v*weight_multiplier for k,v in weight_dict.items()}
        # building criterion
        criterion = SetCriterion(
            enc_cfg['NUM_CLASSES'],
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            grounding_weight=None,
            dn=dec_cfg['DN'],
            dn_losses=dn_losses,
            panoptic_on=dec_cfg['PANO_BOX_LOSS'],
            semantic_ce_loss=dec_cfg['TEST']['SEMANTIC_ON'] and dec_cfg['SEMANTIC_CE_LOSS'] and not dec_cfg['TEST']['PANOPTIC_ON'],
        )

        # build model
        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        # lang_encoder = build_language_encoder(cfg)
        sem_seg_head = build_openseed_head(cfg, backbone.output_shape(), None, extra=extra)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['COCO']['TEST']['DETECTIONS_PER_IMAGE'],
            "data_loader": None,
            "focus_on_box": cfg['MODEL']['DECODER']['TEST']['TEST_FOUCUS_ON_BOX'],
            "transform_eval": cfg['MODEL']['DECODER']['TEST']['PANO_TRANSFORM_EVAL'],
            "pano_temp": cfg['MODEL']['DECODER']['TEST']['PANO_TEMPERATURE'],
            "semantic_ce_loss": cfg['MODEL']['DECODER']['TEST']['SEMANTIC_ON'] and cfg['MODEL']['DECODER']['SEMANTIC_CE_LOSS'] and not cfg['MODEL']['DECODER']['TEST']['PANOPTIC_ON'],
            "train_dataset_name": cfg['DATASETS']['TRAIN'], # HACK for only two training set
            "background": cfg['MODEL'].get('BACKGROUND', True),
            "coco_on": dec_cfg.get('COCO', True),
            "coco_mask_on": dec_cfg.get('COCO_MASK', True),
            "o365_on": dec_cfg.get('O365', True),
            "coco_only": dec_cfg.get('COCO_ONLY', False),
            "detach_seg": cfg.get('detach_seg', False),
            "eval_train": cfg.get('eval_train', False),
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, inference_task='seg'):
        # import ipdb; ipdb.set_trace()
        # print("Num images per batch:",len(batched_inputs['flickr']))
        if self.training:
            losses = {}
            losses_ = dict()
            if 'flickr' in batched_inputs and not self.coco_only:
                self.criterion.conversation=False
                losses_flickr = self.forward_seg(batched_inputs['flickr'], task='seg',default_text_embeddings=batched_inputs['flickr_text_embeddings'],data_type='gd')
                for key, value in losses_flickr.items():
                    losses_['flickr.'+str(key)] = losses_flickr[key]
                self.loss_dict=losses_flickr
            if 'refcoco' in batched_inputs and not self.coco_only:
                self.criterion.conversation=False
                losses_ref = self.forward_seg(batched_inputs['refcoco'], task='seg',default_text_embeddings=batched_inputs['refcoco_text_embeddings'],data_type='ref')
                for key, value in losses_ref.items():
                    losses_['refcoco.'+str(key)] = losses_ref[key]
            if 'vg' in batched_inputs and not self.coco_only:
                self.criterion.conversation = False
                losses_ref = self.forward_seg(batched_inputs['vg'], task='det',
                                              default_text_embeddings=batched_inputs['vg_text_embeddings'],
                                              data_type='ref')
                for key, value in losses_ref.items():
                    losses_['vg.' + str(key)] = losses_ref[key]
                # self.loss_dict=losses_flickr
            if 'coco' in batched_inputs:
                # if self.loss_dict is None:
                #
                # else:
                valid_idx=[]
                for idx,input in enumerate(batched_inputs['coco']):
                    if input['grounding']:
                        valid_idx.append(idx)
                if len(valid_idx)==0:
                    self.criterion.conversation = True
                    losses_flickr = self.forward_seg(batched_inputs['flickr'], task='seg',
                                                     default_text_embeddings=batched_inputs[
                                                         'flickr_text_embeddings'], data_type='coco')
                    self.loss_dict = losses_flickr
                    for key, value in self.loss_dict.items():
                        losses['coco.' + str(key)] = self.loss_dict[key] * 0.0
                else:
                    batched_inputs['coco']=[batched_inputs['coco'][idx] for idx in valid_idx]
                    text_embed=batched_inputs['coco_text_embeddings']
                    text_embed=text_embed[0][valid_idx],text_embed[1][valid_idx]
                    self.criterion.conversation = True
                    losses_coco_instruct = self.forward_seg(batched_inputs['coco'], task='seg',default_text_embeddings=text_embed)
                    for key, value in losses_coco_instruct.items():
                        losses['coco.'+str(key)] = losses_coco_instruct[key]
            losses.update(losses_)
            # if self.task_switch['coco']:
            #     self.criterion.num_classes = 133 if 'pano' in self.train_dataset_name[0] else 80
            #     # self.criterion.num_classes = 133
            #     task = 'seg'
            #     if not self.coco_mask_on:
            #         task = 'det'
            #     # import ipdb; ipdb.set_trace()
            #     losses_coco = self.forward_seg(batched_inputs['coco'], task=task)
            #     new_losses_coco = {}
            #     for key, value in losses_coco.items():
            #         new_losses_coco['coco.'+str(key)] = losses_coco[key]
            #     losses.update(new_losses_coco)
            # if self.task_switch['o365']:
            #     self.criterion.num_classes = 365
            #     losses_o365 = self.forward_seg(batched_inputs['o365'], task='det')
            #     new_losses_o365 = {}
            #     for key, value in losses_o365.items():
            #         new_losses_o365['o365.'+str(key)] = losses_o365[key]
            #     losses.update(new_losses_o365)
            return losses
        else:
            processed_results = self.forward_seg(batched_inputs, task=inference_task)
            return processed_results

    def forward_seg(self, batched_inputs, task='seg',default_text_embeddings=None,data_type='gd'):

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        # features={k:v.to(torch.bfloat16) for k,v in features.items()}
        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images, task=task)
            else:
                targets = None
            outputs, mask_dict = self.sem_seg_head(features, targets=None, task=task,default_text_embeddings=default_text_embeddings)
            ##########eval training
            if self.eval_train:
                pred_logits=outputs["pred_logits"]
                pred_boxes=outputs["pred_boxes"]
                pred_masks=outputs["pred_masks"]>0
                num_total=0
                num_correct=0
                mask_iou=0.0
                # total_union=0.0
                # total_intersection=0.0
                scale_factor=[1024./max(data['height'],data['width']) for data in batched_inputs]
                for i in range(len(pred_logits)):
                    matched_idx=torch.argmax(pred_logits[i],dim=0)
                    matched_boxes=pred_boxes[i][matched_idx]
                    matched_masks=pred_masks[i][matched_idx]
                    gt_boxes_=targets[i]['boxes']
                    gt_masks_=targets[i]['masks']

                    gt_ground_labels=targets[i]['labels']
                    gt_ground_labels_=[]
                    for lb in gt_ground_labels:
                        gt_ground_labels_.extend(lb)
                    max_lb=max(gt_ground_labels_)
                    lb2gt_idx=dict()
                    for lb in range(max_lb+1):
                        lb2gt_idx[lb]=[]
                    for idx,lbs in enumerate(gt_ground_labels):
                        for lb in lbs:
                            lb2gt_idx[lb].append(idx)
                    for lb in range(max_lb+1):
                        pred_box=box_ops.box_cxcywh_to_xyxy(matched_boxes[lb][None])
                        gt_boxes=box_ops.box_cxcywh_to_xyxy(gt_boxes_[lb2gt_idx[lb]])
                        pred_mask=matched_masks[lb]

                        gt_mask=gt_masks_[lb2gt_idx[lb]][0]
                        pred_mask = F.interpolate(
                            pred_mask[None,None].float(),
                            size=(gt_mask.shape[-2], gt_mask.shape[-1]),
                            mode="bilinear",
                            align_corners=False,
                        )[0,0]>0.5
                        if len(gt_boxes)==0:
                            continue
                        mask_iou+=float(torch.sum(pred_mask*gt_mask)/torch.sum(torch.logical_or(pred_mask,gt_mask)))
                        self.total_union+=float(torch.sum(torch.logical_or(pred_mask,gt_mask)))/scale_factor[i]**2
                        self.total_intersection+=float(torch.sum(pred_mask*gt_mask))/scale_factor[i]**2
                        # self.mask_iou+=mask_iou

                        if box_ops.box_iou(pred_box,gt_boxes)[0].max()>0.5:
                            num_correct+=1
                        else:
                            pass
                        num_total+=1
                print(f"{data_type} cIoU:" ,self.total_intersection/self.total_union)
                name_correct='num_correct_'+data_type
                name_total='num_total_'+data_type
                try:
                    gathered_list=[None for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather_object(gathered_list,num_correct)
                    # self.num_correct+=sum(gathered_list)
                    num_correct_value=getattr(self,name_correct)
                    setattr(self,name_correct,num_correct_value+sum(gathered_list))
                    gathered_list=[None for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather_object(gathered_list,num_total)
                    # self.num_total+=sum(gathered_list)
                    num_total_value=getattr(self,name_total)
                    setattr(self,name_total,num_total_value+sum(gathered_list))
                    gathered_list=[None for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather_object(gathered_list,mask_iou)
                    # self.mask_iou+=sum(gathered_list)
                    if torch.distributed.get_rank()==0:
                        print(f"{data_type} acc: ",getattr(self, name_correct) / getattr(self, name_total))
                        # print("mask_iou: ",self.mask_iou/self.num_total)
                except Exception as e:
                    # self.num_correct+=num_correct
                    # self.num_total+=num_total
                    num_correct_value = getattr(self, name_correct)
                    setattr(self, name_correct, num_correct_value + num_correct)
                    num_total_value = getattr(self, name_total)
                    setattr(self, name_total, num_total_value + num_total)
                    try:
                        print(f"{data_type} rank{torch.distributed.get_rank()} acc: ", getattr(self, name_correct) / getattr(self, name_total))
                    except Exception as e:
                        print(f"{data_type} acc: ", getattr(self, name_correct) / getattr(self, name_total))
                ###########################
            # bipartite matching-based loss
            self.criterion.default_text_embeddings = default_text_embeddings
            losses = self.criterion(outputs, targets, mask_dict, task=task)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs, _ = self.sem_seg_head(features)
            mask_cls_results = outputs["pred_logits"]
            mask_box_results = outputs["pred_boxes"]
            if 'seg' in task:
                if task == 'seg':
                    self.semantic_on = self.panoptic_on = self.sem_seg_postprocess_before_inference = self.instance_on = True
                if task == 'inst_seg':
                    self.semantic_on = self.panoptic_on = False
                    self.instance_on = True
                    self.sem_seg_postprocess_before_inference = True
                if task == 'sem_pan_seg':
                    self.semantic_on = self.panoptic_on = True
                    self.instance_on = False
                    self.sem_seg_postprocess_before_inference = True
                if task == 'inst_pan_seg':
                    self.instance_on = self.panoptic_on = True
                    self.semantic_on = False
                    self.sem_seg_postprocess_before_inference = True
                if task == 'sem_seg':
                    self.instance_on = self.panoptic_on = False
                    self.semantic_on = True
                    self.sem_seg_postprocess_before_inference = True
                mask_pred_results = outputs["pred_masks"]
                # upsample masks
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )

            else:
                self.semantic_on = self.panoptic_on = self.sem_seg_postprocess_before_inference = False
                self.instance_on = True
                mask_pred_results = torch.zeros(mask_box_results.shape[0], mask_box_results.shape[1],2, 2).to(mask_box_results)

            del outputs

            processed_results = []

            for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = (images.tensor.shape[-2], images.tensor.shape[-1])  # padded size (divisible to 32)


                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference

                if self.instance_on:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    height = new_size[0]/image_size[0]*height
                    width = new_size[1]/image_size[1]*width
                    mask_box_result = self.box_postprocess(mask_box_result, height, width)

                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                    processed_results[-1]["instances"] = instance_r
            del mask_pred_results
            return processed_results

    def prepare_targets(self, targets, images, task='seg'):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            if task != 'det':
                gt_masks = targets_per_image.gt_masks
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            else:
                padded_masks = None
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes":box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                # print(i, len(keep), lab)
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = Boxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        if self.sem_seg_postprocess_before_inference:
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        else:
            mask_scores_per_image = 1.0
            # labels_per_image = labels_per_image + 1  # HACK for o365 classification
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes

    def forward_eval(self, batched_inputs, text_embeddings):
        # import ipdb; ipdb.set_trace()
        # print("Num images per batch:",len(batched_inputs['flickr']))
        if self.training:
            raise NotImplementedError
        else:
            self.criterion.conversation=False
            box_results, seg_results = self.forward_inner_eval(
                batched_inputs, 
                task='seg',
                default_text_embeddings=text_embeddings,
            )
            return box_results, seg_results

    def forward_inner_eval(self, batched_inputs, task='seg',default_text_embeddings=None):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        matching_threshold = batched_inputs[0]["matching_threshold"] if "matching_threshold" in batched_inputs[0].keys() else None

        features = self.backbone(images.tensor)
        # features={k:v.to(torch.bfloat16) for k,v in features.items()}
        # mask classification target
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images, task=task)
        else:
            targets = None
        default_text_embeddings_ = [default_text_embeddings[0].float(), default_text_embeddings[1]]
        outputs, mask_dict = self.sem_seg_head(features, targets=None, task=task,default_text_embeddings=default_text_embeddings_)
        ##########eval training
        pred_logits=outputs["pred_logits"]
        pred_boxes=outputs["pred_boxes"]
        pred_masks=outputs["pred_masks"]>0
        # scale_factor=[1024./max(data['height'],data['width']) for data in batched_inputs]
        matched_pred_boxes = []
        matched_pred_masks = []
        for i in range(len(pred_logits)):
            if len(pred_logits) > 1:
                raise NotImplementedError
            num_grounding = pred_logits.shape[2]
            for gd_idx in range(num_grounding):
                if matching_threshold is None:
                    matched_idx = torch.argmax(pred_logits[i, :, gd_idx],dim=0)
                    matched_boxes = pred_boxes[i][matched_idx]
                    matched_boxes = matched_boxes[None, :]
                else:
                    matched_idx = torch.where(pred_logits[i, :, gd_idx].softmax(dim=0) > matching_threshold)[0]
                    # print(matched_idx, pred_logits[i, :, gd_idx].softmax(dim=0)[matched_idx])
                    if matched_idx.shape[0] == 0:  #* if there is no one object satisfy threshold, then select the best matched one.
                        matched_boxes = pred_boxes.new_zeros((1, 4))
                        matched_masks = pred_boxes.new_zeros((1, 256, 256))
                    else:
                        matched_boxes = pred_boxes[i][matched_idx]
                        matched_masks = pred_masks[i][matched_idx]
                # matched_masks=pred_masks[i][matched_idx]
                matched_boxes_processed = []
                for lb in range(matched_boxes.shape[0]):
                    pred_box=box_ops.box_cxcywh_to_xyxy(matched_boxes[lb][None])
                    matched_boxes_processed.append(pred_box)
                matched_pred_boxes.append(torch.cat(matched_boxes_processed, dim=0))
                matched_pred_masks.append(matched_masks)
        return matched_pred_boxes, matched_pred_masks
    
@register_model
def get_segmentation_model(cfg, **kwargs):
    return OpenSeeD(cfg)