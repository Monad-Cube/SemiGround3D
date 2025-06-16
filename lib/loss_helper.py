# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from torch.utils import data

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd(), "data")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import MaskedSoftmaxRankingLoss, SoftmaxRankingLoss, SigmoidFocalClassificationLoss, l1_loss, smoothl1_loss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch
#from data.scannet.model_util_scannet import ScannetDatasetConfig


FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
REF_WEIGHTS = [0.1, 0.9]


def compute_kps_loss(data_dict, topk, args):
    box_label_mask = data_dict['box_label_mask']
    seed_inds = data_dict['seed_inds'].long()  # B, K
    seed_xyz = data_dict['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = data_dict['seeds_obj_cls_logits']  # B, 1, K
    gt_center = data_dict['center_label'][:, :, 0:3]  # B, K2, 3
    gt_size = data_dict['size_gts'][:, :, 0:3]  # B, K2, 3
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]
    K2 = gt_center.shape[1]

    point_instance_label = data_dict['point_instance_label']  # B, num_points
    object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
    object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, K2, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
    topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + \
                (box_label_mask[:, :, None] - 1)  # BxK2xtopk
    topk_inds = topk_inds.long()  # BxK2xtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, K2xtopk
    batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous()

    objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    objectness_label[objectness_label_mask < 0] = 0

    total_num_points = B * K
    data_dict[f'points_hard_topk{topk}_pos_ratio'] = \
        torch.sum(objectness_label.float()) / float(total_num_points)
    data_dict[f'points_hard_topk{topk}_neg_ratio'] = 1 - data_dict[f'points_hard_topk{topk}_pos_ratio']

    # Compute objectness loss
    objectness_loss = 0
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
    objectness_loss += cls_loss_src.sum() / B
    if 'kps_ref_score' in data_dict.keys() and args.use_ref_score_loss:
        point_ref_mask = data_dict['point_ref_mask']
        point_ref_mask = torch.gather(point_ref_mask, 1, seed_inds)
        if 'ref_query_points_sample_inds' in data_dict.keys():
            query_points_sample_inds = data_dict['query_points_sample_inds'].long()
            point_ref_mask = torch.gather(point_ref_mask, 1, query_points_sample_inds)
            if args.ref_use_obj_mask: # True
                obj_mask = torch.gather(objectness_label, 1, query_points_sample_inds)
                point_ref_mask = point_ref_mask * obj_mask
        kps_ref_score = data_dict['kps_ref_score']      # [B, 1, N]
        cls_weights = torch.ones((B, kps_ref_score.shape[-1])).cuda().float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        kps_ref_loss = criterion(kps_ref_score.view(kps_ref_score.shape[0], kps_ref_score.shape[2], 1),
                                 point_ref_mask.unsqueeze(-1), weights=cls_weights)
        objectness_loss += kps_ref_loss.sum() / B

    # Compute recall upper bound
    padding_array = torch.arange(0, B).to(point_instance_label.device) * 10000
    padding_array = padding_array.unsqueeze(1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = torch.unique(point_instance_label).shape[0] - 1
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1)
    num_query_bboxes = torch.unique(pos_points_instance_label).shape[0] - 1
    if num_gt_bboxes > 0:
        data_dict[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes

    return objectness_loss, data_dict

def compute_objectness_loss(data_dict, num_decoder_layers, args):
    """ Compute objectness loss for the proposals.
    """

    if num_decoder_layers > 0 and not args.no_detection:
        prefixes = ['proposal_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)] + ['last_']
    else:
        prefixes = ['proposal_']  # only proposal

    objectness_loss_sum = 0.0
    # Associate proposal and GT objects
    seed_inds = data_dict['seed_inds'].long()  # B,num_seed in [0,num_points-1]
    gt_center = data_dict['center_label'][:, :, 0:3]  # B, K2, 3
    query_points_sample_inds = data_dict['query_points_sample_inds'].long()
    B = seed_inds.shape[0]
    K = query_points_sample_inds.shape[1]
    K2 = gt_center.shape[1]
    seed_obj_gt = torch.gather(data_dict['point_obj_mask'], 1, seed_inds)  # B,num_seed
    query_points_obj_gt = torch.gather(seed_obj_gt, 1, query_points_sample_inds)  # B, query_points
    seed_instance_label = torch.gather(data_dict['point_instance_label'], 1, seed_inds)  # B,num_seed
    query_points_instance_label = torch.gather(seed_instance_label, 1, query_points_sample_inds)  # B,query_points
    seed_ref_gt = torch.gather(data_dict['point_ref_mask'], 1, seed_inds)
    query_points_ref_gt = torch.gather(seed_ref_gt, 1, query_points_sample_inds)
    if 'ref_query_points_sample_inds' in data_dict.keys():
        ref_query_points_sample_inds = data_dict['ref_query_points_sample_inds'].long()
        K = ref_query_points_sample_inds.shape[1]
        query_points_obj_gt = torch.gather(query_points_obj_gt, 1, ref_query_points_sample_inds)
        query_points_instance_label = torch.gather(query_points_instance_label, 1, ref_query_points_sample_inds)
        query_points_ref_gt = torch.gather(query_points_ref_gt, 1, ref_query_points_sample_inds)
    
    # Set assignment
    object_assignment = query_points_instance_label  # (B,K) with values in 0,1,...,K2-1
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    # objectness_mask = torch.ones((B, K)).cuda()
    for i, prefix in enumerate(prefixes):
        if i > 0 and f'{prefixes[i-1]}ref_mask_inds' in data_dict.keys():
            ref_mask_inds = data_dict[f'{prefixes[i-1]}ref_mask_inds']
            query_points_obj_gt = torch.gather(query_points_obj_gt, 1, ref_mask_inds)
            # objectness_mask = torch.gather(objectness_mask, 1, ref_mask_inds)
            object_assignment = torch.gather(object_assignment, 1 ,ref_mask_inds)
            query_points_ref_gt = torch.gather(query_points_ref_gt, 1, ref_mask_inds)
        objectness_mask = torch.ones((B, query_points_obj_gt.shape[1])).cuda()
        data_dict[f'{prefix}objectness_label'] = query_points_obj_gt
        data_dict[f'{prefix}objectness_mask'] = objectness_mask
        data_dict[f'{prefix}object_assignment'] = object_assignment
        data_dict[f'{prefix}ref_mask'] = query_points_ref_gt
        total_num_proposal = query_points_obj_gt.shape[0] * query_points_obj_gt.shape[1]
        data_dict[f'{prefix}pos_ratio'] = \
            torch.sum(query_points_obj_gt.float().cuda()) / float(total_num_proposal)
        data_dict[f'{prefix}neg_ratio'] = \
            torch.sum(objectness_mask.float()) / float(total_num_proposal) - data_dict[f'{prefix}pos_ratio']

        # Compute objectness loss
        objectness_scores = data_dict[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = objectness_mask.float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        data_dict[f'{prefix}loss_weights'] = cls_weights
        cls_loss_src = criterion(objectness_scores,
                                 query_points_obj_gt.unsqueeze(-1),
                                 weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B

        data_dict[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss

    return objectness_loss_sum, data_dict

def compute_ref_mask_loss(data_dict, num_decoder_layers, args):
    #prefixes = [f'{i}head_' for i in range(num_decoder_layers - 1)] + ['last_']\
    prefixes = [f'{i}head_' for i in args.ref_filter_steps]
    ref_mask_loss_sum = 0
    B = data_dict['seed_inds'].shape[0]
    for prefix in prefixes:
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = data_dict[f'{prefix}loss_weights']
        ref_mask_label = data_dict[f'{prefix}ref_mask']
        #print(f'{prefix}ref_mask: {torch.sum(ref_mask_label).item()}')
        ref_mask_pre = data_dict[f'{prefix}ref_mask_scores']
        ref_mask_loss = criterion(ref_mask_pre, ref_mask_label.unsqueeze(-1), cls_weights)
        ref_mask_loss = ref_mask_loss.sum() / B
        ref_mask_loss_sum += ref_mask_loss
    return ref_mask_loss_sum / len(prefixes)

def compute_box_and_sem_cls_loss(data_dict, config, num_decoder_layers, args,
                                 center_loss_type='smoothl1', center_delta=1.0,
                                 size_loss_type='smoothl1', size_delta=1.0,
                                 heading_loss_type='smoothl1', heading_delta=1.0,
                                 size_cls_agnostic=False):
    """ Compute 3D bounding box and semantic classification loss.
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    if num_decoder_layers > 0 and not args.no_detection:
        prefixes = ['proposal_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)] + ['last_']
    else:
        prefixes = ['proposal_']  # only proposal
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = data_dict[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]
        # Compute center loss
        pred_center = data_dict[f'{prefix}center']
        gt_center = data_dict['center_label'][:, :, 0:3]

        if center_loss_type == 'smoothl1':
            objectness_label = data_dict[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = data_dict[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute heading loss
        heading_class_label = torch.gather(data_dict['heading_class_label'], 1,
                                           object_assignment)  # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        heading_class_loss = criterion_heading_class(data_dict[f'{prefix}heading_scores'].transpose(2, 1),
                                                     heading_class_label)  # (B,K)
        heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1,
                                              object_assignment)  # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1],
                                                       num_heading_bin).zero_()
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                       1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        heading_residual_normalized_error = torch.sum(
            data_dict[f'{prefix}heading_residuals_normalized'] * heading_label_one_hot,
            -1) - heading_residual_normalized_label


        if heading_loss_type == 'smoothl1':
            heading_residual_normalized_loss = heading_delta * smoothl1_loss(heading_residual_normalized_error,
                                                                             delta=heading_delta)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            heading_residual_normalized_loss = l1_loss(heading_residual_normalized_error)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute size loss
        if size_cls_agnostic:
            pred_size = data_dict[f'{prefix}pred_size']
            size_label = torch.gather(
                data_dict['size_gts'], 1,
                object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
            size_error = pred_size - size_label
            if size_loss_type == 'smoothl1':
                size_loss = size_delta * smoothl1_loss(size_error,
                                                       delta=size_delta)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (
                        torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_loss = l1_loss(size_error)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (
                        torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError
        else:
            size_class_label = torch.gather(data_dict['size_class_label'], 1,
                                            object_assignment)  # select (B,K) from (B,K2)
            criterion_size_class = nn.CrossEntropyLoss(reduction='none')
            size_class_loss = criterion_size_class(data_dict[f'{prefix}size_scores'].transpose(2, 1),
                                                   size_class_label)  # (B,K)
            size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

            size_residual_label = torch.gather(
                data_dict['size_residual_label'], 1,
                object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)

            size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
            size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1),
                                        1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
            size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
            predicted_size_residual_normalized = torch.sum(
                data_dict[f'{prefix}size_residuals_normalized'] * size_label_one_hot_tiled,
                2)  # (B,K,3)

            mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
                0)  # (1,1,num_size_cluster,3)
            mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
            size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)

            size_residual_normalized_error = predicted_size_residual_normalized - size_residual_label_normalized

            if size_loss_type == 'smoothl1':
                size_residual_normalized_loss = size_delta * smoothl1_loss(size_residual_normalized_error,
                                                                           delta=size_delta)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(
                    size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                                                        torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_residual_normalized_loss = l1_loss(size_residual_normalized_error)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(
                    size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                                                        torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError

        # 3.4 Semantic cls loss
        sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(data_dict[f'{prefix}sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        data_dict[f'{prefix}center_loss'] = center_loss
        data_dict[f'{prefix}heading_cls_loss'] = heading_class_loss
        data_dict[f'{prefix}heading_reg_loss'] = heading_residual_normalized_loss
        if size_cls_agnostic:
            data_dict[f'{prefix}size_reg_loss'] = size_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + size_loss
        else:
            data_dict[f'{prefix}size_cls_loss'] = size_class_loss
            data_dict[f'{prefix}size_reg_loss'] = size_residual_normalized_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + 0.1 * size_class_loss + size_residual_normalized_loss
        data_dict[f'{prefix}box_loss'] = box_loss
        data_dict[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss

    return box_loss_sum, sem_cls_loss_sum, data_dict

def compute_reference_loss(data_dict, config, num_decoder_layers, args):
    """ Compute cluster reference loss
    """
    # True
    if args.ref_each_stage:
        prefixes = [f'{i}head_' for i in range(num_decoder_layers - 1)] + ['last_']
    else:
        prefixes = ['last_']  # only proposal
    loss = 0.0
    
    for prefix in prefixes:
        # unpack
        cluster_preds = data_dict[f'{prefix}ref_scores'] # (B, num_proposal)

        # predicted bbox
        pred_ref = data_dict[f'{prefix}ref_scores'].detach().cpu().numpy() # (B, num_proposal)
        pred_center = data_dict[f'{prefix}center'].detach().cpu().numpy() # (B, K, 3)
        pred_heading_class = torch.argmax(data_dict[f'{prefix}heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict[f'{prefix}heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
        pred_size_class = torch.argmax(data_dict[f'{prefix}size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict[f'{prefix}size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class.detach().cpu().numpy()
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

        # ground truth bbox
        gt_center = data_dict['ref_center_label'].cpu().numpy() # (B,3)
        gt_heading_class = data_dict['ref_heading_class_label'].cpu().numpy() # B
        gt_heading_residual = data_dict['ref_heading_residual_label'].cpu().numpy() # B
        gt_size_class = data_dict['ref_size_class_label'].cpu().numpy() # B
        gt_size_residual = data_dict['ref_size_residual_label'].cpu().numpy() # B,3
        # convert gt bbox parameters to bbox corners
        gt_obb_batch = config.param2obb_batch(gt_center[:, 0:3], gt_heading_class, gt_heading_residual,
                        gt_size_class, gt_size_residual)    # B, 7
        gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

        # compute the iou score for all predicted positive ref
        batch_size, num_proposals = cluster_preds.shape
        labels = np.zeros((batch_size, num_proposals))
        for i in range(pred_ref.shape[0]):
            # convert the bbox parameters to bbox corners
            pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                        pred_size_class[i], pred_size_residual[i])
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
            ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[i], (num_proposals, 1, 1)))

            if args.iou_ref_topk:
                if args.iou_ref_th:
                    labels[i, np.intersect1d(np.where(ious > args.iou_ref_th), 
                                            np.argpartition(ious, -args.iou_ref_topk)[-args.iou_ref_topk:])] = 1
                else:
                    labels[i, np.argpartition(ious, -args.iou_ref_topk)[-args.iou_ref_topk:]] = 1
            else:
                labels[i, ious.argmax()] = 1 # treat the bbox with highest iou score as the gt            

        cluster_labels = torch.FloatTensor(labels).cuda()
        # reference loss, ref_criterion is rank
        if args.ref_criterion == 'rank':
            criterion = SoftmaxRankingLoss()
            ref_loss = criterion(cluster_preds, cluster_labels.float().clone())
        else:
            criterion = SigmoidFocalClassificationLoss()
            cls_weights = data_dict[f'{prefix}loss_weights']
            B = cluster_preds.shape[0] 
            ref_loss = criterion(cluster_preds.unsqueeze(-1), cluster_labels.unsqueeze(-1).float().clone(), cls_weights).sum() / B
        loss += ref_loss

    data_dict["cluster_labels"] = cluster_labels
    data_dict['logits'] = cluster_preds
    loss = loss / len(prefixes)

    return loss, data_dict

def compute_lang_classification_loss(data_dict, num_decoder_layers, args):
    if args.ref_each_stage:
        prefixes = [f'{i}head_' for i in range(num_decoder_layers - 1)] + ['last_']
    else:
        prefixes = ['last_']  # only proposal
    loss = 0.0
    for prefix in prefixes:
        criterion = torch.nn.CrossEntropyLoss()
        loss += criterion(data_dict[f'{prefix}lang_logits'], data_dict["object_cat"])
    data_dict['lang_logits'] = data_dict['last_lang_logits']
    loss /= len(prefixes)
    return loss

def get_supervised_loss(data_dict, config, args):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    num_decoder_layers = args.num_decoder_layers
    # KPS loss
    if 'seeds_obj_cls_logits' in data_dict.keys():
        kps_loss, data_dict = compute_kps_loss(data_dict, args.kps_topk, args)
    else:
        kps_loss = torch.zeros(1)[0].cuda()

    # Obj loss
    objectness_loss, data_dict = compute_objectness_loss(data_dict, num_decoder_layers, args)

    # Box loss and sem cls loss
    box_loss, sem_cls_loss, data_dict = compute_box_and_sem_cls_loss(
        data_dict, config, num_decoder_layers, args,
        size_cls_agnostic=args.size_cls_agnostic,
        center_delta=args.center_delta, size_delta=args.size_delta, heading_delta=args.heading_delta,
    )
    # no_detection is false
    if not args.no_detection:
        data_dict['kps_loss'] = kps_loss
        data_dict['objectness_loss'] = objectness_loss
        data_dict['sem_cls_loss'] = sem_cls_loss
        data_dict['box_loss'] = box_loss
    else:
        data_dict['kps_loss'] = torch.zeros(1)[0].cuda()
        data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
        data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['box_loss'] = torch.zeros(1)[0].cuda()
    # no_reference is false
    if not args.no_reference:
        # Reference loss
        ref_loss, data_dict = compute_reference_loss(data_dict, config, num_decoder_layers, args)
        data_dict["ref_loss"] = ref_loss
    else:
        # Reference loss
        objectness_label = data_dict['last_objectness_label']
        data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda()
        data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda()

        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
    # no_lang_cls is false
    if not args.no_lang_cls:
        data_dict["lang_cls_loss"] = compute_lang_classification_loss(data_dict, num_decoder_layers, args)
    else:
        data_dict["lang_cls_loss"] = torch.zeros(1)[0].cuda()
    # use_ref_mask is false
    if args.use_ref_mask:
        data_dict['ref_mask_loss'] = compute_ref_mask_loss(data_dict, num_decoder_layers, args)
    else:
        data_dict['ref_mask_loss'] = torch.zeros(1)[0].cuda()
        
    # Final loss function
    loss = args.kps_loss_weight * data_dict['kps_loss'] \
           + args.det_loss_weight * (0.1*data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1*data_dict['sem_cls_loss']) / (args.num_decoder_layers + 1) \
           + 0.1*data_dict["ref_loss"] \
           + 0.1*data_dict["lang_cls_loss"] \
           + 0.1*data_dict['ref_mask_loss']
    
    loss *= 10 # amplify

    data_dict['loss'] = loss

    return loss, data_dict

def get_loss(data_dict, config, args):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    num_decoder_layers = args.num_decoder_layers
    # KPS loss
    if 'seeds_obj_cls_logits' in data_dict.keys():
        kps_loss, data_dict = compute_kps_loss(data_dict, args.kps_topk, args)
    else:
        kps_loss = torch.zeros(1)[0].cuda()

    # Obj loss
    objectness_loss, data_dict = compute_objectness_loss(data_dict, num_decoder_layers, args)

    # Box loss and sem cls loss
    box_loss, sem_cls_loss, data_dict = compute_box_and_sem_cls_loss(
        data_dict, config, num_decoder_layers, args,
        size_cls_agnostic=args.size_cls_agnostic,
        center_delta=args.center_delta, size_delta=args.size_delta, heading_delta=args.heading_delta,
    )

    if not args.no_detection:
        data_dict['kps_loss'] = kps_loss
        data_dict['objectness_loss'] = objectness_loss
        data_dict['sem_cls_loss'] = sem_cls_loss
        data_dict['box_loss'] = box_loss
    else:
        data_dict['kps_loss'] = torch.zeros(1)[0].cuda()
        data_dict['objectness_loss'] = torch.zeros(1)[0].cuda()
        data_dict['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        data_dict['box_loss'] = torch.zeros(1)[0].cuda()

    if not args.no_reference:
        # Reference loss
        ref_loss, data_dict = compute_reference_loss(data_dict, config, num_decoder_layers, args)
        data_dict["ref_loss"] = ref_loss
    else:
        # Reference loss
        objectness_label = data_dict['last_objectness_label']
        data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda()
        data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape).float().cuda()

        # store
        data_dict["ref_loss"] = torch.zeros(1)[0].cuda()

    if not args.no_lang_cls: # not False = True
        data_dict["lang_cls_loss"] = compute_lang_classification_loss(data_dict, num_decoder_layers, args)
    else:
        data_dict["lang_cls_loss"] = torch.zeros(1)[0].cuda()

    if args.use_ref_mask: # False
        data_dict['ref_mask_loss'] = compute_ref_mask_loss(data_dict, num_decoder_layers, args)
    else:
        data_dict['ref_mask_loss'] = torch.zeros(1)[0].cuda()
        
    # Final loss function
    loss = args.kps_loss_weight * data_dict['kps_loss'] \
           + args.det_loss_weight * (0.1*data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1*data_dict['sem_cls_loss']) / (args.num_decoder_layers + 1) \
           + 0.1*data_dict["ref_loss"] \
           + 0.1*data_dict["lang_cls_loss"] \
           + 0.1*data_dict['ref_mask_loss']
    
    loss *= 10 # amplify

    data_dict['loss'] = loss

    return loss, data_dict

def compute_ref_consistency_loss(data_dict, t_data_dict, config, num_decoder_layers, args):
    """ 
        Compute cluster reference consistency loss
    """
    if args.ref_each_stage:
        prefixes = [f'{i}head_' for i in range(num_decoder_layers - 1)] + ['last_']
    else:
        prefixes = ['last_']  # only proposal
    loss = 0.0

    pseudo_mask = t_data_dict['pseudo_mask'].unsqueeze(1).detach()
    
    for prefix in prefixes:
        # unpack
        cluster_preds = data_dict[f'{prefix}ref_scores'] # (B, num_proposal)

        # student predicted bbox
        pred_ref = data_dict[f'{prefix}ref_scores'].detach().cpu().numpy() # (B,)
        pred_center = data_dict[f'{prefix}center'].detach().cpu().numpy() # (B,K,3)
        pred_heading_class = torch.argmax(data_dict[f'{prefix}heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict[f'{prefix}heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
        pred_size_class = torch.argmax(data_dict[f'{prefix}size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data_dict[f'{prefix}size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_class = pred_size_class.detach().cpu().numpy()
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3
        
        if args.use_each_stage_box:
            # teacher predicted bbox
            t_pred_center = t_data_dict[f'{prefix}center'].detach().cpu().numpy() # (B,K,3)
            t_pred_heading_class = torch.argmax(t_data_dict[f'{prefix}heading_scores'], -1) # B,num_proposal
            t_pred_heading_residual = torch.gather(t_data_dict[f'{prefix}heading_residuals'], 2, t_pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
            t_pred_heading_class = t_pred_heading_class.detach().cpu().numpy() # B,num_proposal
            t_pred_heading_residual = t_pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
            t_pred_size_class = torch.argmax(t_data_dict[f'{prefix}size_scores'], -1) # B,num_proposal
            t_pred_size_residual = torch.gather(t_data_dict[f'{prefix}size_residuals'], 2, t_pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
            t_pred_size_class = t_pred_size_class.detach().cpu().numpy()
            t_pred_size_residual = t_pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3
        else:
            '''
            # watching
            ref_score = t_data_dict['ref_score_plabel']                # (B,)
            ref_soft_score = t_data_dict['ref_soft_score_plabel']      # (B,)
            obj_score = t_data_dict['obj_score_plabel']                # (B,)
            obj_soft_score = t_data_dict['obj_soft_score_plabel']
            size_score = t_data_dict['size_score_plabel']              # (B,)
            size_soft_score = t_data_dict['size_soft_score_plabel'] 
            # ground truth bbox
            gt_center = data_dict['ref_center_label'].cpu().numpy() # (B,3)
            gt_heading_class = data_dict['ref_heading_class_label'].cpu().numpy() # B
            gt_heading_residual = data_dict['ref_heading_residual_label'].cpu().numpy() # B
            gt_size_class = data_dict['ref_size_class_label'].cpu().numpy() # B
            gt_size_residual = data_dict['ref_size_residual_label'].cpu().numpy() # B,3
            '''
            # Pseudo Label bbox
            t_pred_center = t_data_dict['ref_center_plabel'].detach().cpu().numpy() # (B,3)
            t_pred_heading_class = t_data_dict['ref_heading_class_plabel'].detach().cpu().numpy() # B
            t_pred_heading_residual = t_data_dict['ref_heading_residual_plabel'].detach().cpu().numpy() # B
            t_pred_size_class = t_data_dict['ref_size_class_plabel'].detach().cpu().numpy() # B
            t_pred_size_residual = t_data_dict['ref_size_residual_plabel'].detach().cpu().numpy() # B,3

        # t_pred_obb_batch:(centerXYZ, sizeXYZ, heading)
        t_pred_obb_batch = config.param2obb_batch(t_pred_center[:, 0:3], t_pred_heading_class, t_pred_heading_residual,
                t_pred_size_class, t_pred_size_residual)
        # align teacher's predicted center and size based on the input augmentation steps
        if args.use_augment:
            flip_x_axis = data_dict['flip_x_axis'] #(B,)
            flip_y_axis = data_dict['flip_y_axis'] #(B,)
            rot_mat = data_dict['rot_mat']   #(B,3,3)
            scale_ratio = data_dict['scale'].squeeze(1) #(B,1,3) to (B,3)
            # align center
            t_pred = torch.FloatTensor(t_pred_obb_batch).cuda() # to GPU
            inds_to_flip_x_axis = torch.nonzero(flip_x_axis).squeeze(1)
            t_pred[inds_to_flip_x_axis, 0] = -t_pred[inds_to_flip_x_axis, 0]
            inds_to_flip_y_axis = torch.nonzero(flip_y_axis).squeeze(1)
            t_pred[inds_to_flip_y_axis, 1] = -t_pred[inds_to_flip_y_axis, 1]
            t_pred[:, 0:3] = torch.bmm(t_pred[:, 0:3].unsqueeze(1), rot_mat.transpose(1,2)).squeeze(1) # bmm needs 3D Tensor
            t_pred[:, 0:3] = t_pred[:, 0:3] * scale_ratio
            # align size
            t_pred[:, 3:6] = t_pred[:, 3:6] * scale_ratio
            t_pred_obb_batch = t_pred.detach().cpu().numpy()

        t_pred_bbox_batch = get_3d_box_batch(t_pred_obb_batch[:, 3:6], t_pred_obb_batch[:, 6], t_pred_obb_batch[:, 0:3])


        # compute the iou score for all predicted positive ref
        batch_size, num_proposals = cluster_preds.shape # _, num_proposals = 64
        labels = np.zeros((batch_size, num_proposals))

        # batch loop
        for i in range(pred_ref.shape[0]):
            # convert the bbox parameters to bbox corners
            pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i],
                        pred_size_class[i], pred_size_residual[i])
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
            
            # compute ious
            ious = box3d_iou_batch(pred_bbox_batch, np.tile(t_pred_bbox_batch[i], (num_proposals, 1, 1)))
            
            # get topk with iou > threshould, set the labels as 1; iou_ref_topk is 4, iou_ref_threshold is 0.25
            if args.iou_ref_topk:
                if args.iou_ref_th:
                    labels[i, np.intersect1d(np.where(ious > args.iou_ref_th), 
                                            np.argpartition(ious, -args.iou_ref_topk)[-args.iou_ref_topk:])] = 1
                else:
                    labels[i, np.argpartition(ious, -args.iou_ref_topk)[-args.iou_ref_topk:]] = 1
            else:
                labels[i, ious.argmax()] = 1 # treat the bbox with highest iou score as the gt            

        cluster_labels = torch.FloatTensor(labels).cuda() # array 2 tensor
        # reference loss
        if args.ref_criterion == 'rank':
            criterion = MaskedSoftmaxRankingLoss()
            ref_loss = criterion(cluster_preds, cluster_labels.float().clone(), pseudo_mask)
        else:
            criterion = SigmoidFocalClassificationLoss()
            cls_weights = data_dict[f'{prefix}loss_weights']
            B = cluster_preds.shape[0] 
            ref_loss = criterion(cluster_preds.unsqueeze(-1), cluster_labels.unsqueeze(-1).float().clone(), cls_weights).sum() / B
        loss += ref_loss

    data_dict["cluster_labels"] = cluster_labels
    data_dict['logits'] = cluster_preds
    loss = loss / len(prefixes)

    return loss, data_dict

def compute_dks_consistency_loss(data_dict, t_data_dict, topk, args):
    
    # Student's filttered seed points through pointnet
    seed_inds = data_dict['seed_inds'].long()  # (B,1024)
    seed_xyz = data_dict['seed_xyz']  # (B,1024,3)
    seeds_obj_cls_logits = data_dict['seeds_obj_cls_logits']  # (B,1,1024)

    # Pseudo Labels
    #pl_center= t_data_dict["last_center"]       # B,64,3

    # Labels
    gt_center = data_dict['center_label'][:, :, 0:3]  # B, K2, 3
    box_label_mask = data_dict['box_label_mask']    # B, 128
    gt_size = data_dict['size_gts'][:, :, 0:3]  # B, K2, 3
    point_instance_label = data_dict['point_instance_label']  # B, num_points
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]   # 1024
    #K2 = gt_center.shape[1]
    K2 = args.max_num_obj   # K2 = 128
    
    object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
    object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, K2, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)   # (B,1024,128)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
    topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + \
                (box_label_mask[:, :, None] - 1)  # (B,128,TopK), set the point outside bboxes as negative numbers
    topk_inds = topk_inds.long()  # (B,128,TopK)
    topk_inds = topk_inds.view(B, -1).contiguous()  # (B, 128xTopK) = (B,640)
    batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)  # (B,640)
    batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous() # (Bx640,2) [:,0] is batch inds,[:,1] is TopK inds,
    
    # set the points in bboxes as 1, the outside as 0
    objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)    # (B,1024)
    # iterate Bx640 points, set objectness_label[B, 128xTopK]
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    # if point isn't in scene bboxes, set it as 0
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    objectness_label[objectness_label_mask < 0] = 0
    
    total_num_points = B * K
    data_dict[f'points_hard_topk{topk}_pos_ratio'] = \
        torch.sum(objectness_label.float()) / float(total_num_points)
    data_dict[f'points_hard_topk{topk}_neg_ratio'] = 1 - data_dict[f'points_hard_topk{topk}_pos_ratio']

    # Compute objectness loss
    objectness_loss = 0
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
    objectness_loss += cls_loss_src.sum() / B
    if 'kps_ref_score' in data_dict.keys():
        if args.use_ref_score_loss: # labeled data
            point_ref_mask = data_dict['point_ref_mask']
            point_ref_mask = torch.gather(point_ref_mask, 1, seed_inds)
            if 'ref_query_points_sample_inds' in data_dict.keys():
                query_points_sample_inds = data_dict['query_points_sample_inds'].long()
                point_ref_mask = torch.gather(point_ref_mask, 1, query_points_sample_inds) #(B, 768)
                if args.ref_use_obj_mask: # True
                    obj_mask = torch.gather(objectness_label, 1, query_points_sample_inds)
                    point_ref_mask = point_ref_mask * obj_mask
                    kps_ref_score = data_dict['kps_ref_score']      # [B, 1, 768]
            cls_weights = torch.ones((B, kps_ref_score.shape[-1])).cuda().float()
            cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
            cls_weights /= torch.clamp(cls_normalizer, min=1.0)
            kps_ref_loss = criterion(kps_ref_score.view(kps_ref_score.shape[0], kps_ref_score.shape[2], 1),
                                    point_ref_mask.unsqueeze(-1), weights=cls_weights)
            objectness_loss += kps_ref_loss.sum() / B
        if args.use_dks_consistency_loss: # unlabeled data
            kps_ref_score = data_dict['kps_ref_score']      # [B, 1, 768]
            t_kps_ref_score = t_data_dict['kps_ref_score']
            cls_weights = torch.ones((B, kps_ref_score.shape[-1])).cuda().float()
            cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
            cls_weights /= torch.clamp(cls_normalizer, min=1.0)
            kps_ref_loss = criterion(kps_ref_score.view(kps_ref_score.shape[0], kps_ref_score.shape[2], 1),
                                t_kps_ref_score.view(kps_ref_score.shape[0], kps_ref_score.shape[2], 1), weights=cls_weights)
            objectness_loss += kps_ref_loss.sum() / B
    
    # Compute recall upper bound
    padding_array = torch.arange(0, B).to(point_instance_label.device) * 10000
    padding_array = padding_array.unsqueeze(1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = torch.unique(point_instance_label).shape[0] - 1 
    # unique element means the num of GT bboxes, substract background element whose value in label is '-1', 
    # the whole gt bboxes contain all batchsize
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1) # Set the background as '-1', becasuse the instance labels contain '0'
    num_query_bboxes = torch.unique(pos_points_instance_label).shape[0] - 1
    if num_gt_bboxes > 0:
        data_dict[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes

    return objectness_loss, data_dict

def compute_objectness_consistency_loss(data_dict, t_data_dict, num_decoder_layers, args):
    """ Compute objectness loss for the proposals.
    """

    if num_decoder_layers > 0 and not args.no_detection: # True
        prefixes = ['proposal_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)] + ['last_']
    else:
        prefixes = ['proposal_']  # only proposal

    objectness_loss_sum = 0.0
    # Associate proposal and GT objects
    seed_inds = data_dict['seed_inds'].long()  # B,num_seed in [0,num_points-1]
    gt_center = data_dict['center_label'][:, :, 0:3]  # B, K2, 3
    #pl_center = t_data_dict['last'] # (B, 64, 3)
    query_points_sample_inds = data_dict['query_points_sample_inds'].long()
    B = seed_inds.shape[0]
    K = query_points_sample_inds.shape[1] # 768
    K2 = gt_center.shape[1] # 128
    seed_obj_gt = torch.gather(data_dict['point_obj_mask'], 1, seed_inds)  # B,num_seed
    query_points_obj_gt = torch.gather(seed_obj_gt, 1, query_points_sample_inds)  # B, query_points
    seed_instance_label = torch.gather(data_dict['point_instance_label'], 1, seed_inds)  # B,num_seed
    query_points_instance_label = torch.gather(seed_instance_label, 1, query_points_sample_inds)  # B,query_points
    if 'ref_query_points_sample_inds' in data_dict.keys():
        ref_query_points_sample_inds = data_dict['ref_query_points_sample_inds'].long()
        K = ref_query_points_sample_inds.shape[1] # k is updated from 768 to 512
        query_points_obj_gt = torch.gather(query_points_obj_gt, 1, ref_query_points_sample_inds)
        query_points_instance_label = torch.gather(query_points_instance_label, 1, ref_query_points_sample_inds)
        #query_points_ref_gt = torch.gather(query_points_ref_gt, 1, ref_query_points_sample_inds)


    # Set assignment
    object_assignment = query_points_instance_label  # (B,K=512) with values in 0,1,...,K2-1
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    for i, prefix in enumerate(prefixes):
        if i > 0 and f'{prefixes[i-1]}ref_mask_inds' in data_dict.keys():
            ref_mask_inds = data_dict[f'{prefixes[i-1]}ref_mask_inds']
            query_points_obj_gt = torch.gather(query_points_obj_gt, 1, ref_mask_inds)
            object_assignment = torch.gather(object_assignment, 1 ,ref_mask_inds)
        

        objectness_mask = torch.ones((B, query_points_obj_gt.shape[1])).cuda()
        data_dict[f'{prefix}objectness_label'] = query_points_obj_gt
        data_dict[f'{prefix}objectness_mask'] = objectness_mask     # useless
        data_dict[f'{prefix}object_assignment'] = object_assignment # box indx
        total_num_proposal = query_points_obj_gt.shape[0] * query_points_obj_gt.shape[1]
        data_dict[f'{prefix}pos_ratio'] = \
            torch.sum(query_points_obj_gt.float().cuda()) / float(total_num_proposal)
        data_dict[f'{prefix}neg_ratio'] = \
            torch.sum(objectness_mask.float()) / float(total_num_proposal) - data_dict[f'{prefix}pos_ratio']

        # Compute objectness loss
        objectness_scores = data_dict[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = objectness_mask.float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        data_dict[f'{prefix}loss_weights'] = cls_weights
        objectness_loss = 0
        if args.use_obj_consistency_loss:
            cls_loss_src = criterion(objectness_scores,
                                    query_points_obj_gt.unsqueeze(-1),
                                    weights=cls_weights)
            objectness_loss = cls_loss_src.sum() / B

        data_dict[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss

    return objectness_loss_sum, data_dict

def compute_box_and_sem_cls_consistency_loss(data_dict, t_data_dict,config, num_decoder_layers, args,
                                 center_loss_type='smoothl1', center_delta=1.0,
                                 size_loss_type='smoothl1', size_delta=1.0,
                                 heading_loss_type='smoothl1', heading_delta=1.0,
                                 size_cls_agnostic=False):
    """ Compute 3D bounding box and semantic classification loss.
    """

    num_heading_bin = config.num_heading_bin # 1
    num_size_cluster = config.num_size_cluster # 18
    num_class = config.num_class # 18
    mean_size_arr = config.mean_size_arr # 18,3

    if num_decoder_layers > 0 and not args.no_detection: # True
        prefixes = ['proposal_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)] + ['last_']
    else:
        prefixes = ['proposal_']  # only proposal
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = data_dict[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]
        # Compute center loss
        pred_center = data_dict[f'{prefix}center']
        gt_center = data_dict['center_label'][:, :, 0:3]

        if center_loss_type == 'smoothl1':
            objectness_label = data_dict[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = data_dict[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute heading loss
        heading_class_label = torch.gather(data_dict['heading_class_label'], 1,
                                           object_assignment)  # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        heading_class_loss = criterion_heading_class(data_dict[f'{prefix}heading_scores'].transpose(2, 1),
                                                     heading_class_label)  # (B,K)
        heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        heading_residual_label = torch.gather(data_dict['heading_residual_label'], 1,
                                              object_assignment)  # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1],
                                                       num_heading_bin).zero_()
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                       1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        heading_residual_normalized_error = torch.sum(
            data_dict[f'{prefix}heading_residuals_normalized'] * heading_label_one_hot,
            -1) - heading_residual_normalized_label


        if heading_loss_type == 'smoothl1':
            heading_residual_normalized_loss = heading_delta * smoothl1_loss(heading_residual_normalized_error,
                                                                             delta=heading_delta)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            heading_residual_normalized_loss = l1_loss(heading_residual_normalized_error)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute size loss
        if size_cls_agnostic:
            pred_size = data_dict[f'{prefix}pred_size']
            size_label = torch.gather(
                data_dict['size_gts'], 1,
                object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
            size_error = pred_size - size_label
            if size_loss_type == 'smoothl1':
                size_loss = size_delta * smoothl1_loss(size_error,
                                                       delta=size_delta)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (
                        torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_loss = l1_loss(size_error)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (
                        torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError
        else:
            size_class_label = torch.gather(data_dict['size_class_label'], 1,
                                            object_assignment)  # select (B,K) from (B,K2)
            criterion_size_class = nn.CrossEntropyLoss(reduction='none')
            size_class_loss = criterion_size_class(data_dict[f'{prefix}size_scores'].transpose(2, 1),
                                                   size_class_label)  # (B,K)
            size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

            size_residual_label = torch.gather(
                data_dict['size_residual_label'], 1,
                object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)

            size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
            size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1),
                                        1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
            size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
            predicted_size_residual_normalized = torch.sum(
                data_dict[f'{prefix}size_residuals_normalized'] * size_label_one_hot_tiled,
                2)  # (B,K,3)

            mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
                0)  # (1,1,num_size_cluster,3)
            mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
            size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)

            size_residual_normalized_error = predicted_size_residual_normalized - size_residual_label_normalized

            if size_loss_type == 'smoothl1':
                size_residual_normalized_loss = size_delta * smoothl1_loss(size_residual_normalized_error,
                                                                           delta=size_delta)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(
                    size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                                                        torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_residual_normalized_loss = l1_loss(size_residual_normalized_error)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(
                    size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                                                        torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError

        # 3.4 Semantic cls loss
        sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(data_dict[f'{prefix}sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        data_dict[f'{prefix}center_loss'] = center_loss
        data_dict[f'{prefix}heading_cls_loss'] = heading_class_loss
        data_dict[f'{prefix}heading_reg_loss'] = heading_residual_normalized_loss
        if size_cls_agnostic:
            data_dict[f'{prefix}size_reg_loss'] = size_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + size_loss
        else:
            data_dict[f'{prefix}size_cls_loss'] = size_class_loss
            data_dict[f'{prefix}size_reg_loss'] = size_residual_normalized_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + 0.1 * size_class_loss + size_residual_normalized_loss
        data_dict[f'{prefix}box_loss'] = box_loss
        data_dict[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss

    return box_loss_sum, sem_cls_loss_sum, data_dict

def compute_lang_classification_consistency_loss(data_dict, t_data_dict, num_decoder_layers, args):
    if args.ref_each_stage:
        prefixes = [f'{i}head_' for i in range(num_decoder_layers - 1)] + ['last_']
    else:
        prefixes = ['last_']  # only proposal
    loss = 0.0
    for prefix in prefixes:
        criterion = torch.nn.CrossEntropyLoss()
        loss += criterion(data_dict[f'{prefix}lang_logits'], t_data_dict[f'{prefix}lang_logits'])
    data_dict['lang_logits'] = data_dict['last_lang_logits']
    loss /= len(prefixes)
    return loss

def compute_attention_consistency_loss(data_dict, t_data_dict, num_decoder_layers, args):
    B = args.batch_size
    if args.distill_each_stage:
        att_loss_sum = 0
        for i in range(args.num_decoder_layers-1):
            # prefix = 1-3head_, and last_
            prefix = 'last_' if (i == args.num_decoder_layers - 2) else f'{i+1}head_'
            
            # dimension?
            cross_object_att = data_dict[f'{prefix}cross_object_att']
            cross_lang_att = data_dict[f'{prefix}cross_lang_att']
            t_cross_object_att = t_data_dict[f'{prefix}cross_object_att']
            t_cross_lang_att = t_data_dict[f'{prefix}cross_lang_att']
            att_loss = torch.sub(t_cross_object_att, cross_object_att)**2
            result = att_loss.sum()
            att_loss_sum += att_loss.sum() / B #?
        att_consistency_loss = att_loss_sum / (args.num_decoder_layers-1) # .sum()?
    else:
        # distill last layer
        cross_object_att = data_dict['last_cross_object_att']
        cross_lang_att = data_dict['last_cross_lang_att']
        t_cross_object_att = t_data_dict['last_cross_object_att']
        t_cross_lang_att = t_data_dict['last_cross_lang_att']
        att_loss = torch.sub(t_cross_object_att, cross_object_att)**2
        att_consistency_loss = att_loss.sum() / B
    return att_consistency_loss, data_dict

def compute_head_feat_consistency_loss(data_dict, t_data_dict, num_decoder_layers, args):
    B = args.batch_size
    # distill last output feature in predict head
    # cross_object_att = data_dict['last_fused_feat_in_head']
    # t_cross_object_att = t_data_dict['last_fused_feat_in_head']
    
    # Reshape to 2D if needed
    # Assuming cross_object_att is of shape [B, N, D], reshape to [B*N, D]
    #cross_object_att_2d = cross_object_att.reshape(-1, cross_object_att.size(-1))
    #t_cross_object_att_2d = t_cross_object_att.reshape(-1, t_cross_object_att.size(-1))
    
    # Now transpose the 2D tensors
    #t_logits = torch.div(torch.matmul(t_cross_object_att_2d, t_cross_object_att_2d.t()), 0.07)
    #s_logits = torch.div(torch.matmul(cross_object_att_2d, cross_object_att_2d.t()), 0.07)
    
    # att_loss = torch.abs(torch.sub(cross_object_att, t_cross_object_att))
    # att_consistency_loss = att_loss.sum() / B
    cross_object_att = data_dict['last_cross_object_att']
    cross_lang_att = data_dict['last_cross_lang_att']
    t_cross_object_att = t_data_dict['last_cross_object_att']
    t_cross_lang_att = t_data_dict['last_cross_lang_att']
    att_loss = torch.abs(torch.sub(t_cross_object_att, cross_object_att))
    att_consistency_loss = att_loss.sum() / B
    return att_consistency_loss, data_dict

    
def get_consistency_loss(data_dict, t_data_dict, config, args):
    """
    Args:
        end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
                flip_x_axis, flip_y_axis, rot_mat
            }
        ema_end_points: dict
            {
                center, size_scores, size_residuals_normalized, sem_cls_scores,
            }
    Returns:
        consistency_loss: pytorch scalar tensor
        end_points: dict
    """
    num_decoder_layers = args.num_decoder_layers

    # if args.use_dks_consistency_loss is true, dks loss will contain objectness loss and dks consistency loss
    # torch.zero require grad is false
    #

    if args.use_det_loss:
        dks_consistency_loss, data_dict = compute_dks_consistency_loss(data_dict, t_data_dict, args.kps_topk, args)
        obj_consistency_loss, data_dict = compute_objectness_consistency_loss(data_dict, t_data_dict, num_decoder_layers, args)
        box_consistency_loss, sem_cls_consistency_loss, data_dict = compute_box_and_sem_cls_consistency_loss(
            data_dict, t_data_dict, config, num_decoder_layers, args,
            size_cls_agnostic=args.size_cls_agnostic, # False
            center_delta=args.center_delta, size_delta=args.size_delta, heading_delta=args.heading_delta,
        )
    else:
        dks_consistency_loss = torch.zeros(1)[0].cuda()
        obj_consistency_loss = torch.zeros(1)[0].cuda()
        box_consistency_loss = torch.zeros(1)[0].cuda()
        sem_cls_consistency_loss = torch.zeros(1)[0].cuda()

    if args.use_att_consistency_loss:
        #-----OatMatch-----
        att_consistency_loss, data_dict = compute_head_feat_consistency_loss(data_dict, t_data_dict, num_decoder_layers, args)
        #att_consistency_loss, data_dict = compute_attention_consistency_loss(data_dict, t_data_dict, num_decoder_layers, args)
        data_dict['att_consistency_loss'] = att_consistency_loss
    else: 
        att_consistency_loss = torch.zeros(1)[0].cuda()
        data_dict['att_consistency_loss'] = att_consistency_loss

    if args.use_ref_consistency_loss:
        ref_consistency_loss, data_dict = compute_ref_consistency_loss(data_dict, t_data_dict, config, num_decoder_layers, args)
        data_dict['ref_consistency_loss'] = ref_consistency_loss
    else:
        ref_consistency_loss = torch.zeros(1)[0].cuda()
        data_dict['ref_consistency_loss'] = ref_consistency_loss
    #lan_cls_consistency_loss, data_dict = compute_lang_classification_consistency_loss(data_dict, t_data_dict, num_decoder_layers, args)

    consistency_loss = args.kps_loss_weight * dks_consistency_loss \
            + args.det_loss_weight * (0.1*obj_consistency_loss + box_consistency_loss + 0.1*sem_cls_consistency_loss) / (args.num_decoder_layers + 1) \
            + args.ref_loss_weight * ref_consistency_loss \
            + args.att_loss_weight * att_consistency_loss
            #+ 0.1 * lan_cls_consistency_loss
    
    data_dict['dks_consistency_loss'] = dks_consistency_loss
    data_dict['obj_consistency_loss'] = obj_consistency_loss
    data_dict['box_consistency_loss'] = box_consistency_loss
    data_dict['sem_cls_consistency_loss'] = sem_cls_consistency_loss
    data_dict['consistency_loss'] = consistency_loss

    return consistency_loss, data_dict
