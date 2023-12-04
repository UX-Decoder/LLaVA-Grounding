# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from .registry import register_model
from ..utils import configurable, box_ops
from ..backbone import build_backbone, Backbone
from ..body import build_openseed_head
from ..modules import sem_seg_postprocess, HungarianMatcher
from ..modules import SetCriterionLLM as SetCriterion
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
import torch.distributed as dist
import random
import os
import torchvision
from PIL import Image


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        # num_masks,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # only match the lowest loss
    # loss = loss.view(-1, 3)
    # loss = loss.min(1)[0]
    return loss.sum()
    # return loss


def iou_score_loss(inputs, targets):
    ce_loss = F.mse_loss(inputs, targets, reduction="none")
    return ce_loss


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        # num_masks,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.mean(1)
    # loss = loss.view(-1, 3).min(1)[0]

    return loss.sum()
    # return loss


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum()


class SemanticSAM(nn.Module):
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
            ade_on=True,
            merge_class=False,
            sam_on: bool = True,
            pascal_part_on: bool = True,
            regenerate_point: bool = False,
            num_mask_tokens: int = 3,
            interactive_pretrain=False,

            match_loss=True,
            num_vg=2,
            vis_out="vis/",
            coco_old=True,
            clip_on=False,
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
        self.num_vg = num_vg
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss
        self.coco_keys = None
        self.train_class_names = dict()
        self.train_dataset_name = train_dataset_name
        self.coco_mask_on = coco_mask_on
        self.task_switch = {'coco': coco_on, 'o365': o365_on, 'sam': sam_on, 'pascal_part': pascal_part_on,
                            "ade": ade_on}
        self.interactive_pretrain = interactive_pretrain
        self.dbg = False
        self.positive = 0
        self.num_objs = 0
        self.num_hits = 0
        self.num_refer = 0
        self.ref_iou = 0.0
        self.random_iou = 0.0
        self.match_loss = match_loss

        self.clip_on = clip_on
        self.num_all_masks = 0.
        self.coco_old = coco_old
        self.multimodal_cfg = {'is_multimodal': True, 'image_token_len': 140, 'use_im_start_end': True}

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.vis_out = vis_out
        self.obj_projector = nn.Linear(256, 4096)
        print("self.task_switch ", self.task_switch)

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.max_num_instance = 100
        self.num_mask_tokens = num_mask_tokens
        self.regenerate_point = regenerate_point

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']
        
        deep_supervision = dec_cfg['DEEP_SUPERVISION']
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        # loss weights
        iou_weight = dec_cfg['IOU_WEIGHT']
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

        refer_weight = dec_cfg['REFER_WEIGHT']
        fix_backbone = cfg.get('fix_backbone', False)

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
        weight_dict.update({"loss_bbox_0": box_weight, "loss_giou_0": giou_weight})
        weight_dict.update({"iou_score_loss_0": iou_weight})
        weight_dict.update({"loss_mask_part_cls_0": class_weight})
        # two stage is the query selection scheme
        if dec_cfg['TWO_STAGE']:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = dec_cfg['DN']
        # TODO hack for dn lable loss
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k != "loss_mask" and k != "loss_dice"})
            dn_losses = ["dn_labels", "boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses = ["masks", "dn_labels", "boxes"]
        else:
            dn_losses = []
        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k.replace('_0', '_{}'.format(i + 1)): v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if dec_cfg['BOX']:
            losses = ["masks", "labels", "boxes"]
        else:
            losses = ["masks", "labels", ]
        if dec_cfg['PART']:
            losses.append('labels_part')
        weight_dict.update({'all': 1.0, 'sam': 1.0, 'pas': 1.0})

        # update task switch
        task_switch = {}
        task_switch.update({'bbox': dec_cfg.get('DETECTION', True), 'mask': dec_cfg.get('MASK', True)})
        weight_multiplier= dec_cfg.get('WEIGHT_MULTIPLIER', 1.0)
        weight_dict={k:v*weight_multiplier for k,v in weight_dict.items()}

        # building criterion
        criterion = SetCriterion(
            enc_cfg['NUM_CLASSES'],
            matcher=matcher,
            weight_dict=weight_dict,
            # top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            # grounding_weight=None,
            dn=dec_cfg['DN'],
            dn_losses=dn_losses,
            panoptic_on=dec_cfg['PANO_BOX_LOSS'],
            semantic_ce_loss=dec_cfg['TEST']['SEMANTIC_ON'] and dec_cfg['SEMANTIC_CE_LOSS'] and not dec_cfg['TEST'][
                'PANOPTIC_ON'],
            num_mask_tokens=dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3)
        )

        # build model
        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        if fix_backbone:
            for name, param in backbone.named_parameters():
                param.requires_grad = False
        # backbone
        sem_seg_head = build_openseed_head(cfg, backbone.output_shape(), None, extra=extra)
        if fix_backbone:
            for name, param in sem_seg_head.named_parameters():
                param.requires_grad = False

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
            "semantic_ce_loss": cfg['MODEL']['DECODER']['TEST']['SEMANTIC_ON'] and cfg['MODEL']['DECODER'][
                'SEMANTIC_CE_LOSS'] and not cfg['MODEL']['DECODER']['TEST']['PANOPTIC_ON'],
            "train_dataset_name": cfg['DATASETS']['TRAIN'],  # HACK for only two training set
            "background": cfg['MODEL'].get('BACKGROUND', True),
            "coco_on": dec_cfg.get('COCO', True),
            "coco_mask_on": dec_cfg.get('COCO_MASK', True),
            "o365_on": dec_cfg.get('O365', True),
            "sam_on": dec_cfg.get('SAM', True),
            "pascal_part_on": dec_cfg.get('PASCAL', True),
            "regenerate_point": dec_cfg.get('RE_POINT', False),
            "num_mask_tokens": dec_cfg.get('NUM_INTERACTIVE_TOKENS', 3),
            "ade_on": dec_cfg.get('ADE', False),
            "interactive_pretrain": dec_cfg.get('pretrain', False),
            "match_loss": dec_cfg.get('match_loss', True),
            "vis_out": os.path.join(cfg.get('OUTPUT_DIR', 'out'), str(cfg.get('VIS_OUT', 'vis'))),
            "coco_old": cfg.get("coco_old", True),
            "points_per_side_eval": cfg.get("points_per_side_eval", 30),
            "clip_on": cfg.get("clip", False),

        }

    @property
    def device(self):
        return self.pixel_mean.device

    def evaluate_demo(self, batched_inputs, mask_features=None,
                      multi_scale_features=None):
        assert len(batched_inputs) == 1, "only support batch size equal to 1"
        prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        targets = batched_inputs[0]['targets']
        height = images[0].shape[1]
        width = images[0].shape[2]
        padded_h = images.tensor.shape[-2]  # divisable to 32
        padded_w = images.tensor.shape[-1]
        targets[0]['points'] = targets[0]['points'] * torch.as_tensor([width, height, width, height], dtype=torch.float,
                                                                      device=self.device) / torch.as_tensor(
            [padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)

        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(
            features, None)
        outputs, mask_dict = self.sem_seg_head.predictor(multi_scale_features, mask_features, None, targets=targets,
                                                         target_queries=None, target_vlp=None, task='demo',
                                                         extra=prediction_switch)

        pred_ious = None
        if 'pred_ious' in outputs.keys():
            pred_ious = outputs["pred_ious"]

        _, index = pred_ious.view(-1, 3).max(1)
        obj_feats = outputs['obj_features'][0].view(-1, self.num_mask_tokens, 256)
        obj_feats = torch.gather(obj_feats, 1, index[..., None, None].repeat(1, 1, 256))[:, 0]
        mask_pred_results = outputs["pred_masks"]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results.float(),
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        mask_pred_results = mask_pred_results.view(-1, self.num_mask_tokens, images.tensor.shape[-2],
                                                   images.tensor.shape[-1])

        pred_masks = mask_pred_results[:, 0]

        image_size = images.image_sizes[0]

        height = image_size[0]
        width = image_size[1]
        if self.sem_seg_postprocess_before_inference:
            pred_masks = retry_if_cuda_oom(sem_seg_postprocess)(
                pred_masks, image_size, height, width
            )

        return pred_masks, pred_ious, obj_feats

    def forward(self, batched_inputs, inference_task='seg',detach=False):

        assert self.training:
        obj_feats,inter_losses= self.forward_det_pretrain(batched_inputs)
        for k in list(inter_losses.keys()):
            if k in self.criterion.weight_dict:
                inter_losses[k] *= self.criterion.weight_dict[k]
                # losses[k] *= scale
            else:
                # remove this loss if not specified in `weight_dict`
                inter_losses.pop(k)
        new_losses = {}
        for key, value in inter_losses.items():
            new_losses['inter' + '.' + str(key)] = inter_losses[key]
        if detach:
            return [self.obj_projector(feat.detach()) for feat in obj_feats],new_losses
        else:
            return [self.obj_projector(feat)[0] for feat in obj_feats],new_losses

    def forward_det_pretrain(self, batched_inputs, task='seg',
                             prediction_switch={'part': True, 'whole': True, 'seg': True, 'det': True}, dataname='coco',
                             semantic=False):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        mask_features, transformer_encoder_features, multi_scale_features = self.sem_seg_head.pixel_decoder.forward_features(
            features, None)
        if self.clip_on:
            image = \
            preprocess.preprocess(Image.fromarray(batched_inputs[0]['image_ori']), return_tensors='pt')['pixel_values'][
                0]
        prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets_sam(gt_instances, images, prediction_switch=prediction_switch)
        else:
            targets = None
            print("empty targets", targets, task)

        if prediction_switch['whole']:
            prediction_switch['whole'] = False
        if prediction_switch['part']:
            prediction_switch['part'] = False

        obj_features_ls=[]
        losses_total=None
        num_masks=0
        for i,tgt in enumerate(targets):
            tgt_temp=[tgt]
            outputs_gt, mask_dict = self.sem_seg_head.predictor([feat[i:i+1] for feat in multi_scale_features], mask_features[i:i+1], None,
                                                            targets=tgt_temp,
                                                            target_queries=None, target_vlp=None, task='seg',
                                                            extra=prediction_switch)
            self.criterion.index = torch.zeros_like(batched_inputs[i]['instances'].gt_classes).to(outputs_gt['obj_features'].device)

            losses, index = self.criterion(outputs_gt, tgt_temp, mask_dict, task='seg', extra=prediction_switch,
                                           return_idx=True)
            index=self.criterion.index
            bs, n, h, w = outputs_gt["pred_masks"].shape
            obj_features = outputs_gt['obj_features'].view(bs, -1, self.num_mask_tokens, 256)
            obj_features = torch.gather(obj_features, 2, index[None][..., None, None].repeat(1, 1, 1, 256))[:,:, 0]
            # mask_pred_results = outputs_gt["pred_masks"][0].view(-1, self.num_mask_tokens, h, w)

            obj_features_ls.append(obj_features)
            num_masks+=losses['num_masks']
            if losses_total is None:
                losses_total=dict()
                for key in losses.keys():
                    if key != 'num_masks':
                        losses_total[key] = losses[key] * losses['num_masks']
            else:
                for key in losses.keys():
                    if key != 'num_masks':
                        losses_total[key]+=losses[key]*losses['num_masks']
        for key in losses_total.keys():
            if key != 'num_masks':
                losses_total[key] = losses_total[key]/num_masks
        return obj_features_ls,losses_total

    def prepare_targets_sam(self, targets, images, prediction_switch, task='seg', min_box=0.33, max_box=1.0):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        # box_start = random.randint(int((self.max_num_instance - 1)/2), self.max_num_instance - 1)  # box based interactive after this number; about 1/4
        # if random.random()<0.5 and self.dbg:
        #     # import pdb;pdb.set_trace()
        #     targets[0]=targets[0][:0]
        if not self.dbg:
            self.empty_targets = targets
            self.dbg = True
        if len(targets[0]) == 0:
            empty = True
            targets = self.empty_targets
        else:
            empty = False
        for targets_per_image in targets:
            gt_boxes = targets_per_image.gt_boxes if torch.is_tensor(
                targets_per_image.gt_boxes) else targets_per_image.gt_boxes.tensor
            # empty=len(gt_boxes)==0
            assert len(gt_boxes)>0
            self.max_num_instance =  len(gt_boxes)
            box_start = random.randint(int(self.max_num_instance * min_box), int(self.max_num_instance * max_box))
            # pad gt
            h, w = targets_per_image.image_size

            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_masks = targets_per_image.gt_masks if torch.is_tensor(
                targets_per_image.gt_masks) else targets_per_image.gt_masks.tensor

            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            num_mask = targets_per_image.gt_classes.shape[0]

            index = torch.arange(num_mask)

            if self.max_num_instance > num_mask:
                rep = 0 if num_mask == 0 else int(self.max_num_instance / num_mask) + 1
                index = index.repeat(rep)
            index = index[:self.max_num_instance]

            # if self.regenerate_point and box_start > 0:
            point_coords = []
            if box_start > 0:
                for i in range(box_start):
                    mask = gt_masks[index[i]].clone()
                    candidate_indices = mask.nonzero()
                    if len(candidate_indices) == 0:
                        print('wrong')
                        selected_point = torch.tensor([0, 0]).cuda()
                    else:
                        selected_index = random.randint(0, len(candidate_indices) - 1)
                        selected_point = candidate_indices[selected_index].flip(0)
                    selected_point = torch.cat([selected_point - 3, selected_point + 3], 0)
                    point_coords.append(selected_point)
                point_coords = torch.stack(point_coords).to('cuda')
            # else:
                # point_coords = targets_per_image.point_coords[index[:box_start]]
            # point_coords = targets_per_image.gt_boxes.tensor[index[:box_start]]
            new_target = {
                "ori_mask_num": len(targets_per_image.gt_classes),
                "labels": targets_per_image.gt_classes[index] if prediction_switch['whole'] else None,
                "masks": padded_masks[index],
                "boxes": box_ops.box_xyxy_to_cxcywh(gt_boxes[index]) / image_size_xyxy,
                "points": box_ops.box_xyxy_to_cxcywh(point_coords) / image_size_xyxy if len(point_coords) > 0 else None,
                # "pb":torch.randint(2,(min(self.max_num_instance,len(targets_per_image.gt_classes)),),device=gt_masks.device),
                "pb": torch.cat([torch.zeros(box_start), torch.ones(self.max_num_instance - box_start)], 0),
                "gt_whole_classes": targets_per_image.gt_whole_classes[index] if targets_per_image.has(
                    'gt_whole_classes') and prediction_switch['whole'] else None,
                "gt_part_classes": targets_per_image.gt_part_classes[index] if targets_per_image.has(
                    'gt_part_classes') and prediction_switch['part'] else None,
            }
            # handle coco data format
            if prediction_switch['whole'] and not prediction_switch['part']:
                new_target['gt_whole_classes'] = targets_per_image.gt_classes[index]
            if new_target["points"] is not None:
                new_target["boxes_dn"] = torch.cat([new_target["points"], new_target["boxes"][box_start:]], 0)
            else:
                new_target["boxes_dn"] = new_target["boxes"][box_start:]

            new_target['box_start'] = box_start
            new_target['empty'] = empty
            new_targets.append(new_target)

        return new_targets



@register_model
def get_segmentation_model(cfg, **kwargs):
    return SemanticSAM(cfg)