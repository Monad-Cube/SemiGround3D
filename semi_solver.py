'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import json
import copy

from torch.utils.data import dataloader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from collections import OrderedDict
from data.scannet.model_util_scannet import ScannetDatasetConfig

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths, parse_ref_predictions, \
    parse_ref_groundtruths
from lib.config import CONF
from lib.loss_helper import get_supervised_loss, get_consistency_loss, get_loss
from lib.eval_helper import get_eval, get_semi_eval
from lib.pointnet2.pytorch_utils import BNMomentumScheduler
from utils.eta import decode_eta
from utils.pc_utils import write_ply_rgb, write_oriented_bbox
from utils.box_util import get_3d_box, box3d_iou
from scripts.visualize import write_bbox
from scripts.visualize import align_mesh


ITER_REPORT_TEMPLATE = """
-------------------------------burn_in iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_ref_loss: {train_ref_loss}
[loss] train_ref_mask_loss: {train_ref_mask_loss}
[loss] train_lang_cls_loss: {train_lang_cls_loss}
[loss] train_objectness_loss: {train_objectness_loss}
[loss] train_kps_loss: {train_kps_loss}
[loss] train_box_loss: {train_box_loss}
[loss] train_sem_cls_loss: {train_sem_cls_loss}
[loss] train_lang_cls_acc: {train_lang_cls_acc}
[sco.] train_ref_acc: {train_ref_acc}
[sco.] train_obj_acc: {train_obj_acc}
[sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[sco.] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

ITER_REPORT_TEMPLATE_SEMI = """
-------------------------------semi-train iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] consistency_loss: {consistency_loss}
[loss] dks_consistency_loss: {dks_consistency_loss}
[loss] ref_consistency_loss: {ref_consistency_loss}
[loss] obj_consistency_loss: {obj_consistency_loss}
[loss] box_consistency_loss: {box_consistency_loss}
[loss] att_consistency_loss: {att_consistency_loss}
[loss] sem_cls_consistency_loss: {sem_cls_consistency_loss}
[info] filtered_pseudo_labels: {filtered_pseudo_labels}/{iter_id}
[info] filtering_threshold: {filtering_threshold}
[info] mu: {mu}
[info] sigma: {sigma}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_t_forward_time: {mean_t_forward_time}s
[info] mean_s_forward_time: {mean_s_forward_time}s
[info] mean_filtering_time: {mean_filtering_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[train] train_loss: {train_loss}
[train] train_ref_loss: {train_ref_loss}
[train] train_ref_mask_loss: {train_ref_mask_loss}
[train] train_lang_cls_loss: {train_lang_cls_loss}
[train] train_objectness_loss: {train_objectness_loss}
[train] train_kps_loss: {train_kps_loss}
[train] train_box_loss: {train_box_loss}
[train] train_sem_cls_loss: {train_sem_cls_loss}
[train] train_lang_cls_acc: {train_lang_cls_acc}
[train] train_ref_acc: {train_ref_acc}
[train] train_obj_acc: {train_obj_acc}
[train] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
[train] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
[semi_train] dks_consistency_loss: {dks_consistency_loss}
[semi_train] ref_consistency_loss: {ref_consistency_loss}
[semi_train] obj_consistency_loss: {obj_consistency_loss}
[semi_train] box_consistency_loss: {box_consistency_loss}
[semi_train] att_consistency_loss: {att_consistency_loss}
[semi_train] sem_cls_consistency_loss: {sem_cls_consistency_loss}
[semi_train] consistency_loss: {consistency_loss}
[val]   val_loss: {val_loss}
[val]   val_ref_loss: {val_ref_loss}
[val]   val_ref_mask_loss: {val_ref_mask_loss}
[val]   val_lang_cls_loss: {val_lang_cls_loss}
[val]   val_objectness_loss: {val_objectness_loss}
[val]   val_kps_loss: {val_kps_loss}
[val]   val_box_loss: {val_box_loss}
[val]   val_sem_cls_loss: {val_sem_cls_loss}
[val]   val_lang_cls_acc: {val_lang_cls_acc}
[val]   val_ref_acc: {val_ref_acc}
[val]   val_obj_acc: {val_obj_acc}
[val]   val_pos_ratio: {val_pos_ratio}, val_neg_ratio: {val_neg_ratio}
[val]   val_iou_rate_0.25: {val_iou_rate_25}, val_iou_rate_0.5: {val_iou_rate_5}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[loss] loss: {loss}
[loss] ref_loss: {ref_loss}
[loss] ref_mask_loss: {ref_mask_loss}
[loss] lang_cls_loss: {lang_cls_loss}
[loss] objectness_loss: {objectness_loss}
[loss] kps_loss: {kps_loss}
[loss] box_loss: {box_loss}
[loss] sem_cls_loss: {sem_cls_loss}
[loss] lang_cls_acc: {lang_cls_acc}
[sco.] ref_acc: {ref_acc}
[sco.] obj_acc: {obj_acc}
[sco.] pos_ratio: {pos_ratio}, neg_ratio: {neg_ratio}
[sco.] iou_rate_0.25: {iou_rate_25}, iou_rate_0.5: {iou_rate_5}
"""

# constants
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DC = ScannetDatasetConfig()
seed_points_rgb = (0, 127, 255)
sampled_points_rgb = (255, 0, 0)
filter_points_rgb = (255, 128, 0)
selected_points_rgb = (50, 255, 50)
class Solver():
    def __init__(self, student, teacher, data_config, dataloader, scanrefer, optimizer_b, optimizer_s, stamp, val_freq=1, args=None,
    detection=True, reference=True, use_lang_classifier=True,
    lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None, distributed_rank=None):

        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.student = student
        self.teacher = teacher
        self.data_config = data_config
        self.dataloader = dataloader
        self.scanrefer = scanrefer
        self.optimizer_b = optimizer_b
        self.optimizer_s = optimizer_s
        self.stamp = stamp
        self.val_freq = val_freq
        self.args = args

        self.detection = detection
        self.reference = reference
        self.use_lang_classifier = use_lang_classifier
        # pseudo labels
        self.score_threshold = args.score_thr # initialize the threshold
        self.mu = args.mu   
        self.sigma = args.sigma
        self.alpha = args.alpha
        self.alpha_thr = args.alpha_thr
        self.obj_thr = args.obj_thr
        self.size_thr = args.size_thr

        self.lr_decay_step = lr_decay_step # [16, 24, 28]
        self.lr_decay_rate = lr_decay_rate # 0.1
        self.bn_decay_step = bn_decay_step # 10
        self.bn_decay_rate = bn_decay_rate # 0.1


        self.best = {
            "epoch": 0,
            "loss": float("inf"),
            "ref_loss": float("inf"),
            "ref_mask_loss": float("inf"),
            "lang_cls_loss": float("inf"),
            "objectness_loss": float("inf"),
            "kps_loss": float("inf"),
            "box_loss": float("inf"),
            "sem_cls_loss": float("inf"),
            "lang_cls_acc": -float("inf"),
            "ref_acc": -float("inf"),
            "obj_acc": -float("inf"),
            "pos_ratio": -float("inf"),
            "neg_ratio": -float("inf"),
            "iou_rate_0.25": -float("inf"),
            "iou_rate_0.5": -float("inf"),
            "det_mAP_0.25": -float("inf"),
            "det_mAP_0.5": -float("inf")
        }

        # init log
        # contains all necessary info for all phases
        self.log = {
            "burn_in": {},
            "semi_train": {},
            "val": {}
        }
        
        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/burn_in"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/semi_train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "burn_in": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/burn_in")),
            "semi_train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/semi_train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._global_semi_iter_id = 0
        self._filtered_pseudo_labels_iter = 0
        self._reliabel_pseudo_labels_num = 0
        self._global_epoch_id = 0
        self._total_iter = {}             # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__iter_report_template_semi = ITER_REPORT_TEMPLATE_SEMI
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if lr_decay_step and lr_decay_rate:
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer_s, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = StepLR(optimizer_s, lr_decay_step, lr_decay_rate)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            bn_lbmd = lambda it: max(self.args.bn_momentum_init * bn_decay_rate**(int(it / bn_decay_step)), self.args.bn_momentum_min)
            self.bn_scheduler = BNMomentumScheduler(self.student, bn_lambda=bn_lbmd, last_epoch=-1)
        else:
            self.bn_scheduler = None
        
        #if distributed_rank:
        #    nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # EVAL
        # config dict
        self.CONFIG_DICT = {'remove_empty_box': False, 'use_3d_nms': True,
                       'nms_iou': 0.25, 'use_old_type_nms': False, 'cls_nms': True,
                       'per_class_proposal': True, 'conf_thresh': 0.0,
                       'dataset_config': ScannetDatasetConfig()}

        self.AP_IOU_THRESHOLDS = [0.25, 0.5]
        
        # add for distributed
        self.distributed_rank = distributed_rank

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["burn_in"] = len(self.dataloader["burn_in"]) * epoch
        self._total_iter["semi_train"] = len(self.dataloader["semi_train"]) * epoch
        self._total_iter["val"] = len(self.dataloader["val"]) * int(epoch / self.val_freq)
        # Generate Teacher, 
        # TODO, be commentted if training the whole progress
        self._burn_in()

        # 32 epoches
        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))
                
                # feed one epoch
                self._feed(self.dataloader["burn_in"], self.dataloader["semi_train"], "train", epoch_id)
                

                # save model's parameters
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                #torch.save(self.student.state_dict(), os.path.join(model_root, "model_epoch{}.pth".format(epoch_id)))
                torch.save(self.teacher.state_dict(), os.path.join(model_root, "teacher_epoch{}.pth".format(epoch_id)))
                #torch.save(self.teacher.state_dict(), os.path.join(model_root, "model_last.pth"))

                # validation
                self._val(self.dataloader["val"], "val", epoch_id)
                
                
                # update lr scheduler
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                    print("update learning rate --> {}\n".format(self.lr_scheduler.get_lr()))

                # update bn scheduler
                if self.bn_scheduler:
                    self.bn_scheduler.step()
                    print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                # add epoch id
                self._global_epoch_id += 1
                
                
            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        # print and write log once only in main thread, 
        if self.distributed_rank:
            if self.distributed_rank != 0:
                return
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _reset_log(self, phase):
        if phase == "semi_train":
            self.log[phase] = {
                # info
                "t_forward": [],
                "s_forward": [],
                "filtering": [],
                "backward": [],
                "eval": [],
                "fetch": [],
                "iter_time": [],
                # semi-loss (float, not torch.cuda.FloatTensor)
                "dks_consistency_loss": [],
                "ref_consistency_loss": [],
                "obj_consistency_loss": [],
                "box_consistency_loss": [],
                "att_consistency_loss": [],
                "sem_cls_consistency_loss": [],
                "consistency_loss": [],
            }
        else:
            self.log[phase] = {
                # info
                "forward": [],
                "backward": [],
                "eval": [],
                "fetch": [],
                "iter_time": [],
                # loss (float, not torch.cuda.FloatTensor)
                "loss": [],
                "ref_loss": [],
                "ref_mask_loss": [],
                "lang_cls_loss": [],
                "objectness_loss": [],
                "kps_loss": [],
                "box_loss": [],
                "sem_cls_loss": [],
                # semi-loss (float, not torch.cuda.FloatTensor)
                "dks_consistency_loss": [],
                "ref_consistency_loss": [],
                "obj_consistency_loss": [],
                "box_consistency_loss": [],
                "att_consistency_loss": [],
                "sem_cls_consistency_loss": [],
                "consistency_loss": [],
                # scores (float, not torch.cuda.FloatTensor)
                "lang_cls_acc": [],
                "ref_acc": [],
                "obj_acc": [],
                "pos_ratio": [],
                "neg_ratio": [],
                "iou_rate_0.25": [],
                "iou_rate_0.5": []
            }

    def _set_phase(self, phase):
        if phase == "burn_in":
            self.teacher.train()
            self.student.eval()
        elif phase == "semi_train":
            self.teacher.train()
            # cut the grad
            for param in self.teacher.parameters():
                #param.detach_()
                param.requires_grad = False
            #self.teacher.eval()
            self.student.train()
        elif phase == "val":
            self.teacher.eval()
            self.student.eval()
        else:
            raise ValueError("invalid phase")

    def _burn_in(self):
        phase = "burn_in"
        # switch mode
        self._set_phase(phase)

        # re-init log
        self._reset_log(phase)
        # loading model parameter from trained model for semi-training
        if CONF.use_10p_model:
            self._log("Loading 10% trained model state_dict..\n")
            #checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, "5%_data", "checkpoint.tar"))
            #self.teacher.load_state_dict(checkpoint["model_state_dict"])

            path = os.path.join(CONF.PATH.OUTPUT, "10%_data", "model.pth")
            self.teacher.load_state_dict(torch.load(path), strict=True)
    
        # Transfer teacher's model parameters to Student, deepcopy
        self.student.load_state_dict(self.teacher.state_dict())

    def _forward(self, data_dict, stage):
        if stage == "burn_in" or stage == "generate_pesudo_labels": data_dict = self.teacher(data_dict)
        elif stage == "semi_train" or stage == "val": data_dict = self.student(data_dict)
        else:
            raise ValueError("invalid training stage")
        return data_dict

    def _backward(self, lossType=str):
        if lossType == "burn_in":
            # optimize
            self.optimizer_b.zero_grad()
            self._running_log["loss"].backward()
            self.optimizer_b.step()
        elif lossType == "semi_train":
            # optimize
            self.optimizer_s.zero_grad()
            self._running_log["consistency_loss"].backward()
            self.optimizer_s.step()
        else:
            raise ValueError("invalid lossType")
            
    def _update_teacher_via_ema(self, alpha):
        '''
        Update T-Net parameters using EMA
        to prevent the teacher from easily overfitting limited labeled data
        '''
        student_model_dict = self.student.state_dict()
        new_teacher_dict = OrderedDict()
        for key, value in self.teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = value * alpha + student_model_dict[key] * (1 - alpha)
            else:
                raise Exception("{} is not found in student model".format(key))
        self.teacher.load_state_dict(new_teacher_dict) # deepcopy

    def _compute_loss(self, t_data_dict=dict, s_data_dict=dict, lossType = "Grounding_Loss"):
        if lossType == "Grounding_Loss":
            _, t_data_dict = get_supervised_loss(
                data_dict=t_data_dict, 
                config=self.data_config,
                args=self.args,
            )
            # dump
            self._running_log["ref_loss"] = t_data_dict["ref_loss"]
            self._running_log["ref_mask_loss"] = t_data_dict["ref_mask_loss"]
            self._running_log["lang_cls_loss"] = t_data_dict["lang_cls_loss"]
            self._running_log["objectness_loss"] = t_data_dict["objectness_loss"]
            self._running_log["kps_loss"] = t_data_dict["kps_loss"]
            self._running_log["box_loss"] = t_data_dict["box_loss"]
            self._running_log["sem_cls_loss"] = t_data_dict["sem_cls_loss"]
            self._running_log["loss"] = t_data_dict["loss"]

        elif lossType == "Consistency_Loss":
            
            _, s_data_dict = get_consistency_loss(
                data_dict=s_data_dict,
                t_data_dict=t_data_dict, 
                config=self.data_config,
                args=self.args,
            )
            self._running_log["dks_consistency_loss"] = s_data_dict["dks_consistency_loss"]
            self._running_log["ref_consistency_loss"] = s_data_dict["ref_consistency_loss"]
            self._running_log["obj_consistency_loss"] = s_data_dict["obj_consistency_loss"]
            self._running_log["box_consistency_loss"] = s_data_dict["box_consistency_loss"]
            self._running_log["att_consistency_loss"] = s_data_dict["att_consistency_loss"]
            self._running_log["sem_cls_consistency_loss"] = s_data_dict["sem_cls_consistency_loss"]
            self._running_log["consistency_loss"] = s_data_dict["consistency_loss"]
            
            '''
            _, s_data_dict = get_supervised_loss(
                data_dict=s_data_dict, 
                config=self.data_config,
                args=self.args,
            )
            self._running_log["dks_consistency_loss"] = s_data_dict["loss"]
            self._running_log["ref_consistency_loss"] = s_data_dict["loss"]
            self._running_log["obj_consistency_loss"] = s_data_dict["loss"]
            self._running_log["consistency_loss"] = s_data_dict["loss"]
            '''
            
        else:
            raise ValueError("invalid computing loss type")

    def _filter_pseudo_label(self, data_dict) -> bool:
        '''
            Function: filter the unreliable pseudo labels
            Args: data_dict from teacher
            Return: True, filter all the pseudo labels
        '''
        # config, the default no_nms is false
        POST_DICT = {
            'remove_empty_box': True, 
            'use_3d_nms': True, 
            'nms_iou': 0.25,
            'use_old_type_nms': False, 
            'cls_nms': True, 
            'per_class_proposal': True,
            'conf_thresh': 0.05,
            'dataset_config': DC
        }
        skip = False
        use_average = False
        prec = 0
        if(use_average): 
            # get TPM layers
            if self.args.ref_each_stage:
                prefixes = [f'{i}head_' for i in range(self.args.num_decoder_layers - 1)] + ['last_']
            else:
                prefixes = ['last_']  # only proposal
            
            # caculate the reliable point numbers of the proposal
            for prefix in prefixes:
                teacher_ref_scores = data_dict[f'{prefix}ref_scores'] # (B, num_proposal)
                probs = torch.sigmoid(teacher_ref_scores)
                masked_out = torch.masked_select(probs, (probs > 0.7))
                prec += len(masked_out)/teacher_ref_scores.shape[0]/teacher_ref_scores.shape[1]
            prec = prec / len(prefixes)
            if prec < self.args.filtering_precision:
                skip = True
        else:
            # NMS
            if(self.args.use_nms):
                _ = parse_predictions(data_dict, POST_DICT, prefix='last_') # time-consuming
            if 'pred_mask' not in data_dict.keys():
                data_dict['pred_mask'] = torch.ones_like(data_dict['last_objectness_scores']).squeeze(2)
            else:
                data_dict['pred_mask'] = torch.from_numpy(data_dict['pred_mask']).cuda()
            nms_masks = data_dict['pred_mask'].detach().cpu().numpy()

            # Transfer data to (B,num_proposal,...), used for generating one pseudo label
            pred_center = data_dict['last_center'].detach().cpu().numpy()   # B,64,3
            pred_heading_class = torch.argmax(data_dict['last_heading_scores'], -1).detach().cpu().numpy()  # (B,64,1) -> (B,64), scores -> class/0
            pred_heading_residual = data_dict['last_heading_residuals'].squeeze(2).detach().cpu().numpy()  # (B,64,1) -> (B,64)
            pred_size_class = torch.argmax(data_dict['last_size_scores'], -1)     # (B,64,18) -> (B,64), class inds
            pred_size_residual = torch.gather(data_dict['last_size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # (B,64,18,3) -> (B,64,1,3)
            pred_size_class = pred_size_class.detach().cpu().numpy()                  # (B,64)
            pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # (B,64,3)


            pred_objectness = (data_dict['last_objectness_scores'] > 0).squeeze(2).float().detach().cpu().numpy()
            # score
            pred_ref_scores = data_dict["last_ref_scores"].detach().cpu().numpy() #(B,64)
            pred_size_scores = torch.gather(data_dict['last_size_scores'], 2, torch.argmax(data_dict['last_size_scores'], -1).unsqueeze(-1).repeat(1,1,1)).squeeze(-1) #(B,64)
            pred_obj_scores = data_dict['last_objectness_scores'].squeeze(2) #(B,64)
            pred_size_scores_softmax = F.softmax(pred_size_scores, dim=1).detach().cpu().numpy()
            pred_obj_scores_softmax = F.softmax(pred_obj_scores, dim=1).detach().cpu().numpy()
            pred_size_scores = pred_size_scores.detach().cpu().numpy() #(B,64)
            pred_obj_scores = pred_obj_scores.detach().cpu().numpy()

            batch_size = data_dict["last_ref_scores"].shape[0]
            
            # used for selecting one referring proposal
            ref_idx = np.zeros(batch_size) 
            ref_center = np.zeros((batch_size, 3))         # B,3
            ref_heading_class = np.zeros(batch_size)       # B
            ref_heading_residual = np.zeros(batch_size)    # B
            ref_size_class = np.zeros(batch_size)          # B
            ref_size_residual = np.zeros((batch_size, 3))  # B,3
            # score
            ref_score = np.zeros(batch_size)               # B
            obj_score = np.zeros(batch_size)               # B
            size_score = np.zeros(batch_size)              # B
            obj_soft_score = np.zeros(batch_size)          # B
            ref_soft_score = np.zeros(batch_size)          # B
            size_soft_score = np.zeros(batch_size)         # B
            batch_filter_mask = np.zeros(batch_size)       # B
            batch_filter_index = []
            # weight
            pseudo_labels_weight = np.ones(batch_size)     # B

            # caculate score, filter mask and data_dict
            pred_ref_scores_softmax = F.softmax(data_dict["last_ref_scores"] * ( data_dict['last_objectness_scores'] > 0).squeeze(2).float() * data_dict['pred_mask'], dim=1).detach().cpu().numpy()
            for i in range(batch_size):
                pred_masks = nms_masks[i] * pred_objectness[i] == 1
                pred_ref_idx = np.argmax(pred_ref_scores[i] * pred_masks, 0)
                ref_idx[i] = pred_ref_idx
                # select one pseudo label
                ref_center[i] = pred_center[i][pred_ref_idx]
                ref_heading_class[i] = pred_heading_class[i][pred_ref_idx]
                ref_heading_residual[i] = pred_heading_residual[i][pred_ref_idx]
                ref_size_class[i]    = pred_size_class[i][pred_ref_idx]
                ref_size_residual[i] = pred_size_residual[i][pred_ref_idx]
                # scores
                ref_score[i] = pred_ref_scores[i][pred_ref_idx]
                obj_score[i] = pred_obj_scores[i][pred_ref_idx]
                size_score[i] = pred_size_scores[i][pred_ref_idx]
                ref_soft_score[i] = pred_ref_scores_softmax[i][pred_ref_idx]
                obj_soft_score[i] = pred_obj_scores_softmax[i][pred_ref_idx]
                size_soft_score[i] = pred_size_scores_softmax[i][pred_ref_idx]

                # filter mask for loss
                '''
                score = pred_ref_scores_softmax[i, pred_ref_idx]
                if( score > self.score_threshold): 
                    batch_filter_mask[i] = 1
                    batch_filter_index.append(i)
                '''
            # to device
            #ref_idx = torch.IntTensor(ref_idx).cuda()

            if self.args.weighting_mode == "linear_weighting":
                batch_filter_mask[np.where(ref_soft_score>self.score_threshold)] = 1
                if self.args.use_jointly_filtering: # jointly filtering
                    batch_filter_mask[np.where(obj_soft_score < self.obj_thr)] = 0
            elif self.args.weighting_mode == "dynamic_threshold":
                current_thr = np.mean(ref_soft_score)
                self.score_threshold = self.score_threshold * self.alpha_thr + current_thr * (1-self.alpha_thr)
                batch_filter_mask[np.where(ref_soft_score>self.score_threshold)] = 1       
                if self.args.use_jointly_filtering: # jointly filtering
                    batch_filter_mask[np.where(obj_soft_score < self.obj_thr)] = 0
            elif self.args.weighting_mode == "not_weighting":
                batch_filter_mask = ref_soft_score
            elif self.args.weighting_mode == "soft_weighting":
                if self.args.update_mu_sigma: # update mu and sigma via EMA
                    current_mu = np.mean(ref_soft_score)
                    current_sigma = np.var(ref_soft_score)
                    self.mu = self.mu * self.alpha + current_mu * (1-self.alpha)
                    self.sigma = self.sigma * self.alpha + current_sigma * (1-self.alpha) * (batch_size/(batch_size-1))
                # caculate gaussian weight
                weight = np.exp(-((np.clip(ref_soft_score - self.mu, a_min= -1, a_max=0.0) ** 2) / (2 * (self.sigma ** 2))))
                batch_filter_mask = weight
            

            # add to data_dict
            data_dict['ref_score_plabel'] = torch.FloatTensor(ref_score).cuda()                       # (B,)
            data_dict['ref_soft_score_plabel'] = torch.FloatTensor(ref_soft_score).cuda()                  # (B,)
            data_dict['obj_score_plabel'] = torch.FloatTensor(obj_score).cuda()                       # (B,)
            data_dict['obj_soft_score_plabel'] = torch.FloatTensor(obj_soft_score).cuda()
            data_dict['size_score_plabel'] = torch.FloatTensor(size_score).cuda()                     # (B,)
            data_dict['size_soft_score_plabel'] = torch.FloatTensor(size_soft_score).cuda()

            data_dict['ref_center_plabel'] = torch.FloatTensor(ref_center).cuda()                     # (B,64,3)  -> (B,3)
            data_dict['ref_heading_class_plabel'] = torch.IntTensor(ref_heading_class).cuda()         # (B,64,1)  -> B
            data_dict['ref_heading_residual_plabel'] = torch.FloatTensor(ref_heading_residual).cuda() # (B,64,1)  -> B
            data_dict['ref_size_class_plabel'] = torch.IntTensor(ref_size_class).cuda()               # (B,64,18) -> B
            data_dict['ref_size_residual_plabel'] = torch.FloatTensor(ref_size_residual).cuda()       # (B,64,18,3) -> B,3
            data_dict['pseudo_labels_weight'] = torch.FloatTensor(pseudo_labels_weight).cuda()        # (B,)
            data_dict['pseudo_mask'] = torch.IntTensor(batch_filter_mask).cuda()
            #data_dict['pseudo_index'] = torch.IntTensor(batch_filter_index).cuda()
            # Caculation pseudo label/batches numbers
            #reliable_label = np.count_nonzero(batch_filter_mask)
            #self._reliabel_pseudo_labels_num += reliable_label
            # True, while having not reliable pseudo label, then skip this iteration

    def _visualize_pseudo_label(self, args, scanrefer, data, config, split, epoch_id):

             
        # visualize
        dump_dir = os.path.join(CONF.PATH.OUTPUT, self.stamp, "vis", split)
        os.makedirs(dump_dir, exist_ok=True)

        # from inputs
        ids = data['scan_idx'].detach().cpu().numpy()
        point_clouds = data['point_clouds'].cpu().numpy()   # (B, 40000, 9)
        seed_points = data['seed_xyz'].cpu().numpy()
        sampled_points = data['ref_query_points_xyz'].cpu().numpy()
        filter_points = data['last_base_xyz'].cpu().numpy()
        selected_points = data['final_xyz'].cpu().numpy()
        batch_size = point_clouds.shape[0]

        pcl_color = data["pcl_color"].detach().cpu().numpy()
        if args.use_color:
            pcl_color = (pcl_color * 256 + MEAN_COLOR_RGB).astype(np.int64)
        if 'pred_mask' not in data.keys():
            data['pred_mask'] = torch.ones_like(data['last_objectness_scores']).squeeze(2)
        # from network outputs
        # detection
        pred_objectness = (data['last_objectness_scores'] > 0).squeeze(2).float().detach().cpu().numpy() # (B, 64, 1) to (B, 64)
        pred_center = data['last_center'].detach().cpu().numpy() # (B,K,3)
        pred_heading_class = torch.argmax(data['last_heading_scores'], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data['last_heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal
        pred_size_class = torch.argmax(data['last_size_scores'], -1) # B,num_proposal
        pred_size_residual = torch.gather(data['last_size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3
        # reference
        pred_ref_scores = data["last_ref_scores"].detach().cpu().numpy()    # (B, 64)
        pred_ref_scores_softmax = F.softmax(data["last_ref_scores"] * (data['last_objectness_scores'] > 0).squeeze(2).float() * data['pred_mask'], dim=1).detach().cpu().numpy()
        # post-processing
        nms_masks = data['pred_mask'].detach().cpu().numpy() # B,num_proposal
        # ground truth
        gt_center = data['center_label'].cpu().numpy() # (B,MAX_NUM_OBJ,3)
        gt_heading_class = data['heading_class_label'].cpu().numpy() # B,K2
        gt_heading_residual = data['heading_residual_label'].cpu().numpy() # B,K2
        gt_size_class = data['size_class_label'].cpu().numpy() # B,K2
        gt_size_residual = data['size_residual_label'].cpu().numpy() # B,K2,3
        # reference
        gt_ref_labels = data["ref_box_label"].detach().cpu().numpy() # B, 128

        for i in range(batch_size):
            # basic info
            idx = ids[i]
            scene_id = scanrefer[idx]["scene_id"]
            object_id = scanrefer[idx]["object_id"]
            object_name = scanrefer[idx]["object_name"]
            ann_id = scanrefer[idx]["ann_id"]
        
            # scene_output
            scene_dump_dir = os.path.join(dump_dir, scene_id)
            if not os.path.exists(scene_dump_dir):
                os.mkdir(scene_dump_dir)

                # # Dump the original scene point clouds
                mesh = align_mesh(scene_id)
                mesh.write(os.path.join(scene_dump_dir, 'mesh.ply'))

                #write_ply_rgb(point_clouds[i], pcl_color[i], os.path.join(scene_dump_dir, 'pc.ply'))
            # filter out the valid ground truth reference box
            assert gt_ref_labels[i].shape[0] == gt_center[i].shape[0]
            test_labels = gt_ref_labels[i]
            gt_ref_idx = np.argmax(gt_ref_labels[i], 0)

            if split != 'test' and epoch_id < 2:
                # visualize the gt reference box
                # NOTE: for each object there should be only one gt reference box
                object_dump_dir = os.path.join(dump_dir, scene_id, "gt_{}_{}.ply".format(object_id, object_name))
                gt_obb = config.param2obb(gt_center[i, gt_ref_idx, 0:3], gt_heading_class[i, gt_ref_idx], gt_heading_residual[i, gt_ref_idx],
                        gt_size_class[i, gt_ref_idx], gt_size_residual[i, gt_ref_idx])
                gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])

                if not os.path.exists(object_dump_dir):
                    write_bbox(gt_obb, 0, os.path.join(scene_dump_dir, 'gt_{}_{}.ply'.format(object_id, object_name)))

            # find the valid reference prediction
            pred_masks = nms_masks[i] * pred_objectness[i] == 1
            assert pred_ref_scores[i].shape[0] == pred_center[i].shape[0]
            pred_ref_idx = np.argmax(pred_ref_scores[i] * pred_masks, 0)


            proposal_num = pred_center.shape[1]
            for proposal_id in range(proposal_num):
                # visualize the predicted reference box
                pred_obb = config.param2obb(pred_center[i, proposal_id, 0:3], pred_heading_class[i, proposal_id], pred_heading_residual[i, proposal_id],
                        pred_size_class[i, proposal_id], pred_size_residual[i, proposal_id])
                pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
                if split != 'test':
                    iou = box3d_iou(gt_bbox, pred_bbox)
                else:
                    iou = 0
                if(pred_masks[proposal_id]==1):
                    write_bbox(pred_obb, 2, os.path.join(scene_dump_dir, 'obj_{}_pred_{}_{}_{}_{:.5f}_{:.5f}.ply'.format(proposal_id, object_id, object_name, ann_id, pred_ref_scores_softmax[i, proposal_id], iou)))
                else:
                    write_bbox(pred_obb, 1, os.path.join(scene_dump_dir, 'obj_{}_pred_{}_{}_{}_{:.5f}_{:.5f}.ply'.format(proposal_id, object_id, object_name, ann_id, pred_ref_scores_softmax[i, proposal_id], iou)))
            
            seed_points_num = seed_points[i].shape[0]
            write_ply_rgb(seed_points[i], np.array(seed_points_rgb)[np.newaxis, :].repeat(seed_points_num, 0), 
                        os.path.join(scene_dump_dir, 'seed_{}_{}_{}.ply'.format(object_id, object_name, ann_id)))
            sampled_points_num = sampled_points[i].shape[0]
            write_ply_rgb(sampled_points[i], np.array(sampled_points_rgb)[np.newaxis, :].repeat(sampled_points_num, 0), 
                        os.path.join(scene_dump_dir, 'sample_{}_{}_{}.ply'.format(object_id, object_name, ann_id)))
            filter_points_num = filter_points[i].shape[0]
            write_ply_rgb(filter_points[i], np.array(filter_points_rgb)[np.newaxis, :].repeat(filter_points_num, 0), 
                        os.path.join(scene_dump_dir, 'filter_{}_{}_{}.ply'.format(object_id, object_name, ann_id)))    
                
            '''
            # find the valid reference prediction
            pred_masks = nms_masks[i] * pred_objectness[i] == 1
            assert pred_ref_scores[i].shape[0] == pred_center[i].shape[0]
            pred_ref_idx = np.argmax(pred_ref_scores[i] * pred_masks, 0)

            # visualize the predicted reference box
            pred_obb = config.param2obb(pred_center[i, pred_ref_idx, 0:3], pred_heading_class[i, pred_ref_idx], pred_heading_residual[i, pred_ref_idx],
                    pred_size_class[i, pred_ref_idx], pred_size_residual[i, pred_ref_idx])
            pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
            if split != 'test':
                iou = box3d_iou(gt_bbox, pred_bbox)
            else:
                iou = 0

            write_bbox(pred_obb, 1, os.path.join(scene_dump_dir, '{}_pred_{}_{}_{}_{:.5f}_{:.5f}.ply'.format(epoch_id, object_id, object_name, ann_id, pred_ref_scores_softmax[i, pred_ref_idx], iou)))

            seed_points_num = seed_points[i].shape[0]
            write_ply_rgb(seed_points[i], np.array(seed_points_rgb)[np.newaxis, :].repeat(seed_points_num, 0), 
                        os.path.join(scene_dump_dir, '{}_seed_{}_{}_{}_{:.5f}_{:.5f}.ply'.format(epoch_id, object_id, object_name, ann_id, pred_ref_scores_softmax[i, pred_ref_idx], iou)))
            sampled_points_num = sampled_points[i].shape[0]
            write_ply_rgb(sampled_points[i], np.array(sampled_points_rgb)[np.newaxis, :].repeat(sampled_points_num, 0), 
                        os.path.join(scene_dump_dir, '{}_sample_{}_{}_{}_{:.5f}_{:.5f}.ply'.format(epoch_id, object_id, object_name, ann_id, pred_ref_scores_softmax[i, pred_ref_idx], iou)))
            filter_points_num = filter_points[i].shape[0]
            write_ply_rgb(filter_points[i], np.array(filter_points_rgb)[np.newaxis, :].repeat(filter_points_num, 0), 
                        os.path.join(scene_dump_dir, '{}_filter_{}_{}_{}_{:.5f}_{:.5f}.ply'.format(epoch_id, object_id, object_name, ann_id, pred_ref_scores_softmax[i, pred_ref_idx], iou)))
            '''

    def _eval(self, data_dict):
        data_dict = get_eval(
            data_dict=data_dict,
            config=self.data_config,
            reference=self.reference,   # True
            use_lang_classifier=self.use_lang_classifier
        )
        # dump
        self._running_log["lang_cls_acc"] = data_dict["lang_cls_acc"].item()
        self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
        self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        self._running_log["pos_ratio"] = data_dict["last_pos_ratio"].item()
        self._running_log["neg_ratio"] = data_dict["last_neg_ratio"].item()
        self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
        self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])
                
    def _reset_running_log(self, phase):
        if phase == "semi_train":
            self._running_log = {
            # semi loss
            "dks_consistency_loss": 0,
            "ref_consistency_loss": 0,
            "obj_consistency_loss": 0,
            "box_consistency_loss": 0,
            "att_consistency_loss": 0,
            "sem_cls_consistency_loss": 0,
            "consistency_loss": 0,
            }
        else:
            self._running_log = {
                # loss
                "loss": 0,
                "ref_loss": 0,
                "ref_mask_loss": 0,
                "lang_cls_loss": 0,
                "objectness_loss": 0,
                "kps_loss": 0,
                "box_loss": 0,
                "sem_cls_loss": 0,
                # acc
                "lang_cls_acc": 0,
                "ref_acc": 0,
                "obj_acc": 0,
                "pos_ratio": 0,
                "neg_ratio": 0,
                "iou_rate_0.25": 0,
                "iou_rate_0.5": 0
            }

    def _record_data_dict(self, data_dict):
        saved_dict = {}
        saved_dict["last_ref_scores"] = data_dict["last_ref_scores"].detach().cpu().numpy().tolist()
        saved_dict["last_softmax_ref_scores"] = F.softmax(data_dict["last_ref_scores"] * (data_dict['last_objectness_scores'] > 0).squeeze(2).float() * data_dict['pred_mask'], dim=1).detach().cpu().numpy().tolist()
        saved_dict["last_objectness_scores"] = data_dict["last_objectness_scores"].squeeze(2).detach().cpu().numpy().tolist()
        saved_dict["last_heading_scores"] = data_dict["last_size_scores"].detach().cpu().numpy().tolist()
        saved_dict["last_size_scores"] = data_dict["last_size_scores"].detach().cpu().numpy().tolist()
        saved_dict["last_sem_cls_scores"] = data_dict["last_sem_cls_scores"].detach().cpu().numpy().tolist()
        filepath = os.path.join(CONF.PATH.OUTPUT, self.stamp, "score_dict.json")
        json_txt = json.dumps(saved_dict, indent=4)
        with open(filepath, "w") as file: 
            file.write(json_txt)

    def _record_log(self, phase):
        if (not self.args.no_reference) and (phase !="semi_train"):
            self.log[phase]["loss"].append(self._running_log["loss"].item())
            self.log[phase]["ref_loss"].append(self._running_log["ref_loss"].item())
            self.log[phase]["ref_mask_loss"].append(self._running_log["ref_mask_loss"].item())
            self.log[phase]["lang_cls_loss"].append(self._running_log["lang_cls_loss"].item())
            self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
            self.log[phase]["kps_loss"].append(self._running_log["kps_loss"].item())
            self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())
            self.log[phase]["sem_cls_loss"].append(self._running_log["sem_cls_loss"].item())

            self.log[phase]["lang_cls_acc"].append(self._running_log["lang_cls_acc"])
            self.log[phase]["ref_acc"].append(self._running_log["ref_acc"])
            self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
            self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
            self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])
            self.log[phase]["iou_rate_0.25"].append(self._running_log["iou_rate_0.25"])
            self.log[phase]["iou_rate_0.5"].append(self._running_log["iou_rate_0.5"])
        else:
            # NOTE during semi-training, we omit the evaluation on training data
            self.log[phase]["dks_consistency_loss"].append(self._running_log["dks_consistency_loss"].item())
            self.log[phase]["ref_consistency_loss"].append(self._running_log["ref_consistency_loss"].item())
            self.log[phase]["att_consistency_loss"].append(self._running_log["att_consistency_loss"].item())
            self.log[phase]["obj_consistency_loss"].append(self._running_log["obj_consistency_loss"].item())
            self.log[phase]["box_consistency_loss"].append(self._running_log["box_consistency_loss"].item())

            self.log[phase]["sem_cls_consistency_loss"].append(self._running_log["sem_cls_consistency_loss"].item())
            self.log[phase]["consistency_loss"].append(self._running_log["consistency_loss"].item())
    
    def _load_teacher_data_dict(self, data_dict, t_data_dict):
        t_data_dict['point_clouds'] = copy.deepcopy(data_dict['point_clouds'])
        t_data_dict['ema_point_clouds'] = copy.deepcopy(data_dict['ema_point_clouds'])
        t_data_dict["lang_feat"] = copy.deepcopy(data_dict['lang_feat'])
        t_data_dict["lang_mask"] = copy.deepcopy(data_dict['lang_mask'])
        t_data_dict["point_clouds"] = t_data_dict["point_clouds"].cuda()
        t_data_dict["ema_point_clouds"] = t_data_dict["ema_point_clouds"].cuda()
        t_data_dict["lang_feat"] = t_data_dict["lang_feat"].cuda()
        t_data_dict["lang_mask"] = t_data_dict["lang_mask"].cuda()


    def _feed(self, labeled_dataloader, unlabeled_dataloader, phase, epoch_id):
        '''
        #self._log("Loading 10% trained model state_dict..\n")
        #checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, "10%_data_train", "checkpoint.tar"))
        #self.teacher.load_state_dict(checkpoint["model_state_dict"])
        
        self._log("Burning in...\n")
        phase = "burn_in"    
        # switch mode
        self._set_phase(phase)
        # re-init log
        self._reset_log(phase)

        # ---------------- Burn-in ---------------
        for data_dict in labeled_dataloader:
            # move to cuda
            for key in data_dict:
                if key != 'scene_id':
                    data_dict[key] = data_dict[key].cuda()

            # initialize the running loss
            self._reset_running_log(phase)

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())
            
            with torch.autograd.set_detect_anomaly(True):
                # forward
                start = time.time()
                data_dict = self._forward(data_dict, "burn_in") # get grounding result
                self._compute_loss(t_data_dict=data_dict, lossType="Grounding_Loss")
                self.log[phase]["forward"].append(time.time() - start)
                # backward
                start = time.time()
                self._backward("burn_in")
                self.log[phase]["backward"].append(time.time() - start)
            
            # eval on train dataset
            start = time.time()
            if not self.args.no_reference:
                self._eval(data_dict)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            self._record_log(phase)

            # report
            iter_time = self.log[phase]["fetch"][-1]
            iter_time += self.log[phase]["forward"][-1]
            iter_time += self.log[phase]["backward"][-1]
            iter_time += self.log[phase]["eval"][-1]
            self.log[phase]["iter_time"].append(iter_time)
            if (self._global_iter_id + 1) % self.verbose == 0:
                self._train_report(epoch_id, phase)

            # dump log
            self._dump_log(phase)       
            self._global_iter_id += 1   
        
        
        '''
        
        # --------------- semi-train ---------------
        self._log("semi-training start...\n")
        phase = "semi_train"
        # switch mode
        self._set_phase(phase)
        # re-init log
        self._reset_log(phase)

        # data augmentation consume CPU
        for data_dict in unlabeled_dataloader:
            t_data_dict = copy.deepcopy(data_dict)
            #self._load_teacher_data_dict(data_dict, t_data_dict)
            t_data_dict['point_clouds'] = t_data_dict['ema_point_clouds']
            # move to cuda, will it be faster after removing the for-in loop?
            
            for key in data_dict:
                if key != 'scene_id':
                    data_dict[key] = data_dict[key].cuda()
                    t_data_dict[key] = t_data_dict[key].cuda()
            
            # initialize the running loss
            self._reset_running_log(phase)

            # load
            self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())
            start = time.time()
            
            # Filter the unreliable pseudo label when it's not the consistency_loss_ablation
            if not self.args.consistency_loss_ex:
                with torch.no_grad():
                    t_data_dict = self._forward(t_data_dict, "generate_pesudo_labels")
                    self.log[phase]["t_forward"].append(time.time() - start)
                    start = time.time()
                    if self.args.use_ref_consistency_loss:
                        self._filter_pseudo_label(t_data_dict)
                    self.log[phase]["filtering"].append(time.time() - start)
                    start = time.time()

            ################################################
            #                                              #
            # Teacher net has generated the reliable label #
            # Filter part of pesudo labels                 #
            #                                              #
            ################################################
            
            with torch.autograd.set_detect_anomaly(True):
                # HACK edit here
                # forward    
                data_dict = self._forward(data_dict, "semi_train")

                self._compute_loss(t_data_dict=t_data_dict, s_data_dict=data_dict, lossType="Consistency_Loss")
                self.log[phase]["s_forward"].append(time.time() - start)
                # backward
                start = time.time()
                self._backward("semi_train")
                self.log[phase]["backward"].append(time.time() - start)
            
            # NOTE does here updating ema parameters ?

            # update teacher model parameters via ema
            self._update_teacher_via_ema(self.args.alpha)

            # eval student on train dataset,
            # omit it during semi-training
            start = time.time()
            #if not self.args.no_reference:
            #    self._eval(data_dict)
            # visualize pseudo labels
            #if(batch_idx==0):
                #self._visualize_pseudo_label(self.args, self.scanrefer["semi_train"], data_dict, DC, "semi-train", epoch_id)
            self.log[phase]["eval"].append(time.time() - start)

            # record log
            self._record_log(phase)

            # report
            iter_time = self.log[phase]["fetch"][-1]
            iter_time += self.log[phase]["s_forward"][-1]
            iter_time += self.log[phase]["t_forward"][-1]
            iter_time += self.log[phase]["filtering"][-1]
            iter_time += self.log[phase]["backward"][-1]
            iter_time += self.log[phase]["eval"][-1]
            self.log[phase]["iter_time"].append(iter_time)
            if (self._global_semi_iter_id + 1) % self.verbose == 0:
                self._train_report(epoch_id, phase)

            # dump log
            self._dump_log(phase)
            
            self._global_semi_iter_id += 1

        # update threshold
        #if self.args.use_incremental_thr:
        #    self.score_threshold += 0.03
        

    def _val(self, dataloader, phase, epoch_id):
        # evaluation
        print("evaluating...")
        # switch mode
        self._set_phase(phase)  # val
        # re-init log
        self._reset_log(phase)
        # change dataloader, what is the meaning of this step??
        dataloader = tqdm(dataloader)

        for data_dict in dataloader:
            self._reset_running_log(phase)
            # move to cuda
            for key in data_dict:
                if key != 'scene_id':
                    data_dict[key] = data_dict[key].cuda()
            # with no need to backward        
            with torch.no_grad():
                # forward
                data_dict = self._forward(data_dict, "val")
                self._compute_loss(data_dict, "Grounding_Loss")
                # NOTE after semi-training, the _eval function will get the necessary data, 
                # such as data_dict["last_objectness_label"], and we omit _eval during semi-training
                if not self.args.no_reference:
                    self._eval(data_dict)
                self._record_log(phase)
            
        # test mAP
        if self.args.eval_det:
            self.evaluate_detection_one_epoch(self.dataloader["val"], self.args)
        if self.args.eval_ref:
            self.evaluate_reference_one_epoch(self.dataloader["val"], self.args)
        
        self._dump_log("val")
        self._epoch_report(epoch_id)
        
        if self.args.no_reference:
            cur_criterion = 'det_mAP_0.5'
        else:
            cur_criterion = "iou_rate_0.5"
        cur_best = np.mean(self.log[phase][cur_criterion])
        if cur_best > self.best[cur_criterion]:
            self._log("best {} achieved: {}".format(cur_criterion, cur_best))
            self._log("current burn_in_loss: {}".format(np.mean(self.log["burn_in"]["loss"])))
            #self._log("current semi_train_consistency_loss: {}".format(np.mean(self.log["semi_train"]["consistency_loss"])))
            self._log("current val_loss: {}".format(np.mean(self.log["val"]["loss"])))
            self.best["epoch"] = epoch_id + 1
            self.best["loss"] = np.mean(self.log[phase]["loss"])
            self.best["ref_loss"] = np.mean(self.log[phase]["ref_loss"])
            self.best["ref_mask_loss"] = np.mean(self.log[phase]["ref_mask_loss"])
            self.best["lang_cls_loss"] = np.mean(self.log[phase]["lang_cls_loss"])
            self.best["objectness_loss"] = np.mean(self.log[phase]["objectness_loss"])
            self.best["kps_loss"] = np.mean(self.log[phase]["kps_loss"])
            self.best["box_loss"] = np.mean(self.log[phase]["box_loss"])
            self.best["sem_cls_loss"] = np.mean(self.log[phase]["sem_cls_loss"])
            self.best["lang_cls_acc"] = np.mean(self.log[phase]["lang_cls_acc"])
            self.best["ref_acc"] = np.mean(self.log[phase]["ref_acc"])
            self.best["obj_acc"] = np.mean(self.log[phase]["obj_acc"])
            self.best["pos_ratio"] = np.mean(self.log[phase]["pos_ratio"])
            self.best["neg_ratio"] = np.mean(self.log[phase]["neg_ratio"])
            self.best["iou_rate_0.25"] = np.mean(self.log[phase]["iou_rate_0.25"])
            self.best["iou_rate_0.5"] = np.mean(self.log[phase]["iou_rate_0.5"])
            self.best["det_mAP_0.25"] = np.mean(self.log[phase]["det_mAP_0.25"])
            self.best["det_mAP_0.5"] = np.mean(self.log[phase]["det_mAP_0.5"])

            # save model
            self._log("saving best models...\n")
            model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
            torch.save(self.student.state_dict(), os.path.join(model_root, "model.pth"))

    def _dump_log(self, phase):
        log = {
            "burn_in": {
                "loss": ["loss", "ref_loss", "ref_mask_loss", "lang_cls_loss", "objectness_loss", "kps_loss", "box_loss", "sem_cls_loss"],
                "score": ["lang_cls_acc", "ref_acc", "obj_acc", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5"]
            },
            "semi_train": {
                "loss": ["consistency_loss","ref_consistency_loss","dks_consistency_loss","obj_consistency_loss","box_consistency_loss","att_consistency_loss","sem_cls_consistency_loss"],
            },
            'val': {
                "loss": ["loss", "ref_loss", "ref_mask_loss", "lang_cls_loss", "objectness_loss", "kps_loss", "box_loss", "sem_cls_loss"],
                "score": ["lang_cls_acc", "ref_acc", "obj_acc", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5", "det_mAP_0.25", "det_mAP_0.5"]
            }
        }
        index = {
            "burn_in": self._global_iter_id,
            "semi_train": self._global_semi_iter_id,
            "val": self._global_iter_id + self._global_semi_iter_id,
        }
        if self.distributed_rank:
            if self.distributed_rank != 0:
                return
        for key in log[phase]:
            for item in log[phase][key]:
                self._log_writer[phase].add_scalar(
                    "{}/{}".format(key, item),
                    np.mean([v for v in self.log[phase][item]]),
                    index[phase]
                )

    def _finish(self, epoch_id):
        # print best
        self._best_report()
        '''
        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.student.state_dict(),
            "optimizer_b_state_dict": self.optimizer_b.state_dict(),
            "optimizer_s_state_dict": self.optimizer_s.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))
        '''
        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.student.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["burn_in", "semi_train", "val"]:
            self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id, phase):
        # compute ETA
        if phase == "semi_train":
            t_forward_time = self.log[phase]["t_forward"]
            s_forward_time = self.log[phase]["s_forward"]
            filtring_time = self.log[phase]["filtering"]
            forward_time = self.log[phase]["s_forward"]
        else:   
            forward_time = self.log[phase]["forward"]
        
        fetch_time = self.log[phase]["fetch"]
        backward_time = self.log[phase]["backward"]
        eval_time = self.log[phase]["eval"]
        iter_time = self.log[phase]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
            

        # print report
        if phase == "burn_in":
            eta_sec = (self._total_iter[phase] - self._global_iter_id - 1) * mean_train_time
            eta_sec += (self._total_iter["semi_train"] - self._global_semi_iter_id-1) * mean_est_val_time
            eta_sec += (self._total_iter["val"] - int(self._global_epoch_id / self.val_freq * len(self.dataloader["val"]))) * mean_est_val_time
            eta = decode_eta(eta_sec)
            
            iter_report = self.__iter_report_template.format(
                epoch_id=epoch_id + 1,
                iter_id=self._global_iter_id + 1,
                total_iter=self._total_iter[phase],
                train_loss=round(np.mean([v for v in self.log[phase]["loss"]]), 5),
                train_ref_loss=round(np.mean([v for v in self.log[phase]["ref_loss"]]), 5),
                train_ref_mask_loss=round(np.mean([v for v in self.log[phase]["ref_mask_loss"]]), 5),
                train_lang_cls_loss=round(np.mean([v for v in self.log[phase]["lang_cls_loss"]]), 5),
                train_objectness_loss=round(np.mean([v for v in self.log[phase]["objectness_loss"]]), 5),
                train_kps_loss=round(np.mean([v for v in self.log[phase]["kps_loss"]]), 5),
                train_box_loss=round(np.mean([v for v in self.log[phase]["box_loss"]]), 5),
                train_sem_cls_loss=round(np.mean([v for v in self.log[phase]["sem_cls_loss"]]), 5),
                train_lang_cls_acc=round(np.mean([v for v in self.log[phase]["lang_cls_acc"]]), 5),
                train_ref_acc=round(np.mean([v for v in self.log[phase]["ref_acc"]]), 5),
                train_obj_acc=round(np.mean([v for v in self.log[phase]["obj_acc"]]), 5),
                train_pos_ratio=round(np.mean([v for v in self.log[phase]["pos_ratio"]]), 5),
                train_neg_ratio=round(np.mean([v for v in self.log[phase]["neg_ratio"]]), 5),
                train_iou_rate_25=round(np.mean([v for v in self.log[phase]["iou_rate_0.25"]]), 5),
                train_iou_rate_5=round(np.mean([v for v in self.log[phase]["iou_rate_0.5"]]), 5),
                mean_fetch_time=round(np.mean(fetch_time), 5),
                mean_forward_time=round(np.mean(forward_time), 5),
                mean_backward_time=round(np.mean(backward_time), 5),
                mean_eval_time=round(np.mean(eval_time), 5),
                mean_iter_time=round(np.mean(iter_time), 5),
                eta_h=eta["h"],
                eta_m=eta["m"],
                eta_s=eta["s"]
            )
        elif phase == "semi_train":
            eta_sec = (self._total_iter[phase] - self._global_semi_iter_id - 1) * mean_train_time
            eta_sec += (self._total_iter["val"] - int(self._global_epoch_id / self.val_freq * len(self.dataloader["val"]))) * mean_est_val_time
            eta = decode_eta(eta_sec)

            iter_report = self.__iter_report_template_semi.format(
                epoch_id=epoch_id + 1,
                iter_id=self._global_semi_iter_id + 1,
                total_iter=self._total_iter[phase],
                filtered_pseudo_labels=self._filtered_pseudo_labels_iter,
                filtering_threshold=self.score_threshold,
                mu=self.mu,
                sigma=self.sigma,
                consistency_loss=round(np.mean([v for v in self.log[phase]["consistency_loss"]]), 5),
                ref_consistency_loss=round(np.mean([v for v in self.log[phase]["ref_consistency_loss"]]), 5),
                dks_consistency_loss=round(np.mean([v for v in self.log[phase]["dks_consistency_loss"]]), 5),
                obj_consistency_loss=round(np.mean([v for v in self.log[phase]["obj_consistency_loss"]]), 5),
                box_consistency_loss=round(np.mean([v for v in self.log[phase]["box_consistency_loss"]]), 5),
                att_consistency_loss=round(np.mean([v for v in self.log[phase]["att_consistency_loss"]]), 5),
                sem_cls_consistency_loss=round(np.mean([v for v in self.log[phase]["sem_cls_consistency_loss"]]), 5),
                mean_fetch_time=round(np.mean(fetch_time), 5),
                mean_t_forward_time=round(np.mean(t_forward_time), 5),
                mean_s_forward_time=round(np.mean(s_forward_time), 5),
                mean_filtering_time=round(np.mean(filtring_time), 5),
                mean_backward_time=round(np.mean(backward_time), 5),
                mean_eval_time=round(np.mean(eval_time), 5),
                mean_iter_time=round(np.mean(iter_time), 5),
                eta_h=eta["h"],
                eta_m=eta["m"],
                eta_s=eta["s"]
            )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            train_loss=round(np.mean([v for v in self.log["burn_in"]["loss"]]), 5),
            train_ref_loss=round(np.mean([v for v in self.log["burn_in"]["ref_loss"]]), 5),
            train_ref_mask_loss=round(np.mean([v for v in self.log["burn_in"]["ref_mask_loss"]]), 5),
            train_lang_cls_loss=round(np.mean([v for v in self.log["burn_in"]["lang_cls_loss"]]), 5),
            train_objectness_loss=round(np.mean([v for v in self.log["burn_in"]["objectness_loss"]]), 5),
            train_kps_loss=round(np.mean([v for v in self.log["burn_in"]["kps_loss"]]), 5),
            train_box_loss=round(np.mean([v for v in self.log["burn_in"]["box_loss"]]), 5),
            train_sem_cls_loss=round(np.mean([v for v in self.log["burn_in"]["sem_cls_loss"]]), 5),
            train_lang_cls_acc=round(np.mean([v for v in self.log["burn_in"]["lang_cls_acc"]]), 5),
            train_ref_acc=round(np.mean([v for v in self.log["burn_in"]["ref_acc"]]), 5),
            train_obj_acc=round(np.mean([v for v in self.log["burn_in"]["obj_acc"]]), 5),
            train_pos_ratio=round(np.mean([v for v in self.log["burn_in"]["pos_ratio"]]), 5),
            train_neg_ratio=round(np.mean([v for v in self.log["burn_in"]["neg_ratio"]]), 5),
            train_iou_rate_25=round(np.mean([v for v in self.log["burn_in"]["iou_rate_0.25"]]), 5),
            train_iou_rate_5=round(np.mean([v for v in self.log["burn_in"]["iou_rate_0.5"]]), 5),
            dks_consistency_loss=round(np.mean([v for v in self.log["burn_in"]["dks_consistency_loss"]]), 5),
            ref_consistency_loss=round(np.mean([v for v in self.log["burn_in"]["ref_consistency_loss"]]), 5),
            obj_consistency_loss=round(np.mean([v for v in self.log["burn_in"]["obj_consistency_loss"]]), 5),
            box_consistency_loss=round(np.mean([v for v in self.log["burn_in"]["box_consistency_loss"]]), 5),
            att_consistency_loss=round(np.mean([v for v in self.log["burn_in"]["att_consistency_loss"]]), 5),
            sem_cls_consistency_loss=round(np.mean([v for v in self.log["burn_in"]["sem_cls_consistency_loss"]]), 5),
            consistency_loss=round(np.mean([v for v in self.log["burn_in"]["consistency_loss"]]), 5),
            val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
            val_ref_loss=round(np.mean([v for v in self.log["val"]["ref_loss"]]), 5),
            val_ref_mask_loss=round(np.mean([v for v in self.log["val"]["ref_mask_loss"]]), 5),
            val_lang_cls_loss=round(np.mean([v for v in self.log["val"]["lang_cls_loss"]]), 5),
            val_objectness_loss=round(np.mean([v for v in self.log["val"]["objectness_loss"]]), 5),
            val_kps_loss=round(np.mean([v for v in self.log["val"]["kps_loss"]]), 5),
            val_box_loss=round(np.mean([v for v in self.log["val"]["box_loss"]]), 5),
            val_sem_cls_loss=round(np.mean([v for v in self.log["val"]["sem_cls_loss"]]), 5),
            val_lang_cls_acc=round(np.mean([v for v in self.log["val"]["lang_cls_acc"]]), 5),
            val_ref_acc=round(np.mean([v for v in self.log["val"]["ref_acc"]]), 5),
            val_obj_acc=round(np.mean([v for v in self.log["val"]["obj_acc"]]), 5),
            val_pos_ratio=round(np.mean([v for v in self.log["val"]["pos_ratio"]]), 5),
            val_neg_ratio=round(np.mean([v for v in self.log["val"]["neg_ratio"]]), 5),
            val_iou_rate_25=round(np.mean([v for v in self.log["val"]["iou_rate_0.25"]]), 5),
            val_iou_rate_5=round(np.mean([v for v in self.log["val"]["iou_rate_0.5"]]), 5),
        )
        self._log(epoch_report)
    
    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            loss=round(self.best["loss"], 5),
            ref_loss=round(self.best["ref_loss"], 5),
            ref_mask_loss=round(self.best["ref_mask_loss"], 5),
            lang_cls_loss=round(self.best["lang_cls_loss"], 5),
            objectness_loss=round(self.best["objectness_loss"], 5),
            kps_loss=round(self.best["kps_loss"], 5),
            box_loss=round(self.best["box_loss"], 5),
            sem_cls_loss=round(self.best["sem_cls_loss"], 5),
            lang_cls_acc=round(self.best["lang_cls_acc"], 5),
            ref_acc=round(self.best["ref_acc"], 5),
            obj_acc=round(self.best["obj_acc"], 5),
            pos_ratio=round(self.best["pos_ratio"], 5),
            neg_ratio=round(self.best["neg_ratio"], 5),
            iou_rate_25=round(self.best["iou_rate_0.25"], 5),
            iou_rate_5=round(self.best["iou_rate_0.5"], 5),
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)

    def evaluate_detection_one_epoch(self, test_loader, config):
        self._log('=====================>DETECTION EVAL<=====================')
        stat_dict = {}

        if config.num_decoder_layers > 0:
            prefixes = ['last_', 'proposal_'] + [f'{i}head_' for i in range(config.num_decoder_layers - 1)]
        else:
            prefixes = ['proposal_']  # only proposal
        ap_calculator_list = [APCalculator(iou_thresh, self.data_config.class2type) \
                              for iou_thresh in self.AP_IOU_THRESHOLDS]
        mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in self.AP_IOU_THRESHOLDS]

        self.student.eval()  # set model to eval mode (for bn and dp)
        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        scene_set = set()

        for batch_idx, batch_data_label in enumerate(test_loader):
            scene_idx_list = []
            for i, scene_id in enumerate(batch_data_label['scene_id']):
                if scene_id not in scene_set:
                    scene_set.add(scene_id)
                    scene_idx_list.append(i)

            if len(scene_idx_list) == 0:
                continue

            for key in batch_data_label:
                if key != 'scene_id':
                    batch_data_label[key] = batch_data_label[key][scene_idx_list, ...].cuda(non_blocking=True)

            # Forward pass
            with torch.no_grad():
                data_dict = self.student(batch_data_label)

            # Compute loss
            for key in batch_data_label:
                if key in data_dict:
                    continue
                data_dict[key] = batch_data_label[key]
            loss, data_dict = get_supervised_loss(data_dict, self.data_config, self.args)

            # Accumulate statistics and print out
            for key in data_dict:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if 'weights' in key:    # add for not save loss_weight
                        continue
                    if key not in stat_dict: stat_dict[key] = 0
                    if isinstance(data_dict[key], float):
                        stat_dict[key] += data_dict[key]
                    else:
                        stat_dict[key] += data_dict[key].item()

            for prefix in prefixes:
                batch_pred_map_cls = parse_predictions(data_dict, self.CONFIG_DICT, prefix,
                                                       size_cls_agnostic=config.size_cls_agnostic)
                batch_gt_map_cls = parse_groundtruths(data_dict, self.CONFIG_DICT,
                                                      size_cls_agnostic=config.size_cls_agnostic)
                batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
                batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        for prefix in prefixes:
            for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                              batch_gt_map_cls_dict[prefix]):
                for ap_calculator in ap_calculator_list:
                    ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
            # Evaluate average precision
            for i, ap_calculator in enumerate(ap_calculator_list):
                metrics_dict = ap_calculator.compute_metrics()
                self._log(f'=====================>{prefix} IOU THRESH: {self.AP_IOU_THRESHOLDS[i]}<=====================')
                for key in metrics_dict:
                    self._log(f'{key} {metrics_dict[key]}')
                if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                    mAP = metrics_dict['mAP']
                mAPs[i][1][prefix] = metrics_dict['mAP']
                ap_calculator.reset()

        for mAP in mAPs:
            self._log(
                f'IoU[{mAP[0]}]:\t' + ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))
        # add map to log
        for mAP in mAPs:
            self.log['val']["det_mAP_{}".format(mAP[0])] = [mAP[1]['last_']]
        return mAP, mAPs

    def evaluate_reference_one_epoch(self, test_loader, config):
        self._log('=====================>REFERENCE EVAL<=====================')
        stat_dict = {}

        if config.num_decoder_layers > 0:
            prefixes = ['last_']
        else:
            return
        ap_calculator_list = [APCalculator(iou_thresh, self.data_config.class2type) \
                              for iou_thresh in self.AP_IOU_THRESHOLDS]
        mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in self.AP_IOU_THRESHOLDS]

        self.student.eval()  # set model to eval mode (for bn and dp)
        batch_pred_map_cls_dict = {k: [] for k in prefixes}
        batch_gt_map_cls_dict = {k: [] for k in prefixes}

        for batch_idx, batch_data_label in enumerate(test_loader):
            for key in batch_data_label:
                if key != 'scene_id':
                    batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

            # Forward pass
            with torch.no_grad():
                data_dict = self.student(batch_data_label)

            # Compute loss
            for key in batch_data_label:
                if key in data_dict:
                    continue
                data_dict[key] = batch_data_label[key]
            loss, data_dict = get_supervised_loss(data_dict, self.data_config, self.args)
            
            # Accumulate statistics and print out
            for key in data_dict:
                if 'loss' in key or 'acc' in key or 'ratio' in key:
                    if 'weights' in key:    # add for not save loss_weight
                        continue
                    if key not in stat_dict: stat_dict[key] = 0
                    if isinstance(data_dict[key], float):
                        stat_dict[key] += data_dict[key]
                    else:
                        stat_dict[key] += data_dict[key].item()

            for prefix in prefixes:
                batch_pred_map_cls = parse_ref_predictions(data_dict, self.CONFIG_DICT, prefix,
                                                       size_cls_agnostic=config.size_cls_agnostic)
                batch_gt_map_cls = parse_ref_groundtruths(data_dict, self.CONFIG_DICT,
                                                      size_cls_agnostic=config.size_cls_agnostic)
                batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
                batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

        mAP = 0.0
        for prefix in prefixes:
            for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                              batch_gt_map_cls_dict[prefix]):
                for ap_calculator in ap_calculator_list:
                    ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
            # Evaluate average precision
            for i, ap_calculator in enumerate(ap_calculator_list):
                metrics_dict = ap_calculator.compute_metrics()
                self._log(f'=====================>{prefix} IOU THRESH: {self.AP_IOU_THRESHOLDS[i]}<=====================')
                for key in metrics_dict:
                    self._log(f'{key} {metrics_dict[key]}')
                if prefix == 'last_' and ap_calculator.ap_iou_thresh > 0.3:
                    mAP = metrics_dict['mAP']
                mAPs[i][1][prefix] = metrics_dict['mAP']
                ap_calculator.reset()

        for mAP in mAPs:
            self._log(
                f'IoU[{mAP[0]}]:\t' + ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))

        return mAP, mAPs