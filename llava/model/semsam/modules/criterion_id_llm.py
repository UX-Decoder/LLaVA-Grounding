# ------------------------------------------------------------------------
# Copyright (c) IDEA, Inc. and its affiliates.
# Modified from DINO https://github.com/IDEA-Research/DINO by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from ..utils import box_ops
from utils.utils import slprint

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
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
    return loss
    # return loss.mean(1).sum() / num_boxes


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
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
    # return loss.sum() / num_masks
    return loss


def iou_score_loss(inputs, targets):
    ce_loss = F.mse_loss(inputs, targets, reduction="none")
    return ce_loss


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
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

    # return loss.sum() / num_masks
    return loss


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


class SetCriterionLLM(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, dn="no", dn_losses=[], panoptic_on=False,
                 semantic_ce_loss=False, num_mask_tokens=3, iou_loss=True):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_classes_part = -1
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.dn = dn
        self.dn_losses = dn_losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25

        self.panoptic_on = panoptic_on
        self.semantic_ce_loss = semantic_ce_loss
        self.num_mask_tokens = num_mask_tokens
        self.index = None
        self.iou_loss = iou_loss
        self.prediction_switch = None
        self.index_switch = {'part': torch.arange(0, self.num_mask_tokens - 1).cuda(),
                             'whole': torch.arange(self.num_mask_tokens - 1, self.num_mask_tokens).cuda(),
                             'all': torch.arange(0, self.num_mask_tokens).cuda(), }
        # self.dbg_f=open("/comp_robot/zhanghao/model/idino_llama_coco/dbg","a")
        self.keys="loss_bbox_0, loss_giou_0, loss_mask_bce_0, loss_mask_dice_0, iou_score_loss_0, loss_bbox_1, loss_giou_1, loss_mask_bce_1, loss_mask_dice_1, iou_score_loss_1, loss_bbox_2, loss_giou_2, loss_mask_bce_2, loss_mask_dice_2, iou_score_loss_2, loss_bbox_3, loss_giou_3, loss_mask_bce_3, loss_mask_dice_3, iou_score_loss_3, loss_bbox_4, loss_giou_4, loss_mask_bce_4, loss_mask_dice_4, iou_score_loss_4, loss_bbox_5, loss_giou_5, loss_mask_bce_5, loss_mask_dice_5, iou_score_loss_5, loss_bbox_6, loss_giou_6, loss_mask_bce_6, loss_mask_dice_6, iou_score_loss_6, loss_bbox_7, loss_giou_7, loss_mask_bce_7, loss_mask_dice_7, iou_score_loss_7, loss_bbox_8, loss_giou_8, loss_mask_bce_8, loss_mask_dice_8, iou_score_loss_8"
        self.keys=self.keys.split(", ")
        print("iou_loss is ", iou_loss)

    def loss_labels_ce(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True, key='gt_whole_classes'):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits' in outputs
        if self.prediction_switch is None or 'whole' not in self.prediction_switch.keys():
            if 'labels' in targets[0].keys() and targets[0]['labels'] is not None:
                key = 'labels'
        else:
            if not self.prediction_switch['whole']:
                return {"fake_no_loss_mask_cls_0": 0.0}
            elif key not in targets[0].keys():
                # FIXME only consider batchsize=1 case
                assert len(targets) == 1
                return {"loss_mask_cls_0": 0.0 * outputs['pred_logits'].sum()}
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[key][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {}
        loss_ce = loss_ce.sum(2)
        losses["loss_mask_cls_0"] = torch.gather(loss_ce.view(-1, 3), 1, self.index.unsqueeze(1)).mean().sum() / num_boxes
        # losses = {"loss_mask_cls_0": loss_ce}
        # losses={k:losses[k].to(torch.bfloat16) for k in losses.keys()}
        return losses

    def loss_labels_part(self, outputs, targets, indices, num_boxes, log=True, key='gt_part_classes'):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits_part' in outputs
        if not self.prediction_switch['part']:
            return {"fake_no_loss_mask_part_cls_0": 0.0}
        elif key not in targets[0].keys():
            # FIXME only consider batchsize=1 case
            assert len(targets) == 1
            # return {"loss_mask_whole_cls_0": 0.0*outputs['pred_logits_part'].sum()}
            return {"loss_mask_part_cls_0": 0.0 * outputs['pred_logits_part'].sum()}
        src_logits = outputs['pred_logits_part']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[key][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes_part,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * \
                  src_logits.shape[1]
        losses = {}
        loss_ce = loss_ce.sum(2)
        losses["loss_mask_part_cls_0"] = torch.gather(loss_ce.view(-1, 3), 1, self.index.unsqueeze(1)).mean().sum() / num_boxes
        # losses = {"loss_mask_part_cls_0": loss_ce}

        return losses

    def loss_boxes_o365(self, outputs, targets, indices, num_boxes, layer_id=None, extra=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        if indices is None or len(targets) == 0:
            loss = outputs['pred_boxes'].sum() * 0.0
            losses = {"loss_bbox_0": loss, "loss_giou_0": loss}
            return losses

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox_0'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou_0'] = loss_giou.sum() / num_boxes

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        if 'boxes' not in targets[0].keys():
            # FIXME only consider batchsize=1 case
            assert len(targets) == 1
            return {"loss_bbox_0": 0.0 * outputs['pred_boxes'].sum(),
                    "loss_giou_0": 0.0 * outputs['pred_boxes'].sum(), }
        assert self.index is not None
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # print(src_boxes)
        # print(target_boxes)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        loss_bbox = loss_bbox.sum(1)
        # losses["loss_bbox_0"] = loss_bbox.sum() / num_boxes
        try:
            losses["loss_bbox_0"] = torch.gather(loss_bbox.view(-1, 3), 1, self.index.unsqueeze(1)).sum() / num_boxes

            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
            losses["loss_giou_0"] = torch.gather(loss_giou.view(-1, 3), 1, self.index.unsqueeze(1)).sum() / num_boxes
        except:
            losses["loss_bbox_0"] = loss_bbox.sum()*0.0
            losses["loss_giou_0"] = loss_bbox.sum()*0.0
            print(loss_bbox.view(-1, 3))
            print(self.index.unsqueeze(1))
        # losses={k:losses[k].to(torch.bfloat16) for k in losses.keys()}

        return losses

    def loss_boxes_panoptic(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        isthing = target_labels < 80
        target_boxes = target_boxes[isthing]
        src_boxes = src_boxes[isthing]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses["loss_bbox_0"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_giou_0"] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        if 'masks' not in targets[0].keys():
            # FIXME only consider batchsize=1 case
            assert len(targets) == 1
            return {"loss_mask_bce_0": 0.0 * outputs['pred_masks'].sum(),
                    "loss_mask_dice_0": 0.0 * outputs['pred_masks'].sum(),
                    "iou_score_loss_0": 0.0 * outputs['pred_masks'].sum(),
                    }

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        # import pdb;pdb.set_trace()
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.float(),
                lambda logits: calculate_uncertainty(logits.float()),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks.float(),
                point_coords.float(),
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks.float(),
            point_coords.float(),
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            # "loss_mask_bce_0": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_mask_dice_0": dice_loss_jit(point_logits, point_labels, num_masks),
            # "loss_mask_dice_0": dice_loss(point_logits, point_labels, num_masks),
        }
        mask_loss = losses["loss_mask_bce_0"] + losses["loss_mask_dice_0"]
        mask_loss, index = mask_loss.view(-1, 3).min(1)
        # slprint(index)
        # if len(targets)>1:   # FIXME starting box index is the same
        #     assert targets[0]['box_start'] == targets[1]['box_start']
        bs = outputs["pred_masks"].shape[0]
        box_start = targets[0]['box_start']
        # index.view(bs, -1)[:, box_start:] = 0   # all the box index is set to 0
        if self.index is None:
            self.index = index
        else:
            index=self.index
        losses["loss_mask_bce_0"] = torch.gather(losses["loss_mask_bce_0"].view(-1, 3), 1,
                                                 index.unsqueeze(1)).sum() / num_masks
        dice_loss = torch.gather(losses["loss_mask_dice_0"].view(-1, 3), 1, index.unsqueeze(1))
        losses["loss_mask_dice_0"] = dice_loss.sum() / num_masks

        target_iou = 1 - dice_loss
        src_ious = outputs["pred_ious"]
        iou_idx = ([src_idx[0].view(bs, -1)[:, :src_ious.shape[1]].flatten(),
                    src_idx[1].view(bs, -1)[:, :src_ious.shape[1]].flatten()])
        # print("loss_masks1")
        # slprint(target_iou)
        # print("loss_masks2")
        #
        # slprint(src_ious)
        # print("loss_masks3")
        #
        # slprint(iou_idx)
        # print("loss_masks4")
        #
        # slprint(index.unsqueeze(1))
        src_ious = src_ious[iou_idx]
        src_ious = torch.gather(src_ious, 1, index.unsqueeze(1))
        #
        # if self.iou_loss:
        losses['iou_score_loss_0'] = iou_score_loss(src_ious, target_iou).sum() / num_masks
        # losses={k:losses[k].to(torch.bfloat16) for k in losses.keys()}
        del src_masks
        del target_masks
        return losses

    def loss_labels_o365(self, outputs, targets, indices, num_boxes, log=True, layer_id=None, extra=None):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        if indices is None or len(targets) == 0:
            loss_ce = outputs['pred_logits'].sum() * 0.0
            losses = {"loss_mask_cls_0": loss_ce}
            return losses

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2).mean(1).sum()
        losses = {'loss_mask_cls_0': loss_ce}

        return losses

    def prep_for_dn(self, mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        known_indice = mask_dict['known_indice']
        scalar, pad_size = mask_dict['scalar'], mask_dict['pad_size']
        assert pad_size % scalar == 0
        single_pad = pad_size // scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes, num_tgt, single_pad, scalar

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            'labels_o365':  self.loss_labels_o365,
            'labels_part': self.loss_labels_part,
            'masks': self.loss_masks,
            'boxes': self.loss_boxes_panoptic if self.panoptic_on else self.loss_boxes,
            'boxes_o365': self.loss_boxes_o365,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, mask_dict=None, task='sam', extra={}, return_idx=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        # if self.dn is not "no" and mask_dict is not None:
        #     output_known_lbs_bboxes,num_tgt,single_pad,scalar = self.prep_for_dn(mask_dict)
        assert len(targets)==1, "now only support one image training for interactive segmentation"
        prediction_switch = extra
        self.prediction_switch = prediction_switch

        exc_idx = []
        key = 'pred_boxes'
        for i in range(len(targets)):
            if len(targets[i]['boxes']) > 0:
                if task=='det':
                    tgt_idx = torch.arange(0, len(targets[i]['boxes'])).long().cuda()
                else:
                    tgt_idx = torch.arange(0, len(targets[i]['boxes'])).long().cuda().repeat_interleave(
                        self.num_mask_tokens)
                src_idx = torch.arange(0, outputs[key].shape[1]).long().cuda()
                # tgt_idx = t.flatten()
                # output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                # output_idx = output_idx.flatten()
            else:
                output_idx = tgt_idx = src_idx = torch.tensor([]).long().cuda()
            exc_idx.append((src_idx, tgt_idx))
        indices = exc_idx
        # indices = self.matcher(outputs_without_aux, targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["boxes"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=outputs[key].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        losses['num_masks'] = num_masks
        if 'masks' in 'masks':
            assert 'masks' in self.losses[0], "must calculate mask loss first for match"
        # slprint(outputs)
        # slprint(targets)
        # slprint(indices)
        for loss in self.losses:
            if task=='det':
                if loss=='labels_part':
                    continue
                if  loss=='labels':
                    loss='labels_o365'
                if  loss=='boxes':
                    loss='boxes_o365'
                if loss == 'masks':
                    l_dict = dict()
                    l_dict['loss_mask_bce_0'] = torch.as_tensor(0.).to('cuda')
                    l_dict['loss_mask_dice_0'] = torch.as_tensor(0.).to('cuda')
                    losses.update(l_dict)
            else:
                if 'labels' in loss:
                    continue
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))
        index=self.index
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if task == 'det':
                        if loss == 'labels_part':
                            continue
                        if loss == 'labels':
                            loss = 'labels_o365'
                        if loss == 'boxes':
                            loss = 'boxes_o365'
                        if loss=='masks':
                            l_dict=dict()
                            l_dict['loss_mask_bce_0'] = torch.as_tensor(0.).to('cuda')
                            l_dict['loss_mask_dice_0'] = torch.as_tensor(0.).to('cuda')
                            l_dict = {k.replace('_0', f"_{i + 1}"): v for k, v in l_dict.items()}
                            losses.update(l_dict)
                    else:
                        if 'labels' in loss:
                            continue
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                        # l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        l_dict = {k.replace('_0', f"_{i + 1}"): v for k, v in l_dict.items()}
                        losses.update(l_dict)
        # totoal_loss = torch.tensor(0.0).cuda()
        # for k,v in losses.items():
        #     totoal_loss += v
        # losses = dict()
        # losses['all'] = totoal_loss
        # assert "iou_score_loss_0" in losses, losses.keys()
        # self.dbg_f.write(", ".join(list(losses.keys()))+'\n')
        losses={k:losses[k] for k in losses.keys()}
        if return_idx:
            return losses,index
        else:
            return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
