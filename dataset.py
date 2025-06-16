'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp
from torch.utils.data import Dataset
import clip
import copy

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")
#train on 36665 samples and val on 9508 samples
from lib.config import CONF
class ScannetReferenceDataset(Dataset):
       
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False, 
        lang_emb_type='glove'):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.lang_emb_type = lang_emb_type

        # load data
        self._load_data()
        self.multiview_data = {}
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        if self.lang_emb_type == 'clip' and CONF.word_erase > 0 and (self.split == 'burn_in' or self.split == 'semi_train'):
            lang_len = len(lang_feat)
            choice_idx = np.random.choice(lang_len, int(float(CONF.word_erase) * lang_len))
            # breakpoint()
            for i in choice_idx:
                lang_feat[i] = ""
            # lang_feat[choice_idx] = 'UNK'
            lang_token = lang_feat
            lang_feat = clip.tokenize(' '.join(lang_feat), truncate=True).squeeze(0)
            lang_len = lang_feat.argmax().item() + 1
        if self.lang_emb_type == 'glove':
            lang_token = self.scanrefer[idx]["token"]
            lang_len = len(self.scanrefer[idx]["token"])
            lang_len = lang_len if lang_len <= CONF.max_des_len else CONF.max_des_len
        elif self.lang_emb_type == 'clip':
            lang_token = self.scanrefer[idx]["token"]
            lang_len = lang_feat.argmax().item() + 1
        else:
            raise NotImplementedError('Language embedding type unsupported!')
        lang_mask = np.zeros((CONF.max_des_len, ))
        lang_mask[:lang_len] = 1.0
        if CONF.use_context_label:
            context_label = set()
            for word in lang_token:
                if word in DC.type2class.keys() and word != 'others':
                    context_label.add(DC.type2class[word])

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]     # (50000,9), (xyz,color,normal)
        instance_labels = self.scene_data[scene_id]["instance_labels"] # (50000)
        semantic_labels = self.scene_data[scene_id]["semantic_labels"] # (50000)
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"] # scene_id=562, (21,8)

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
            #ema_pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
            #ema_pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            if CONF.fuse_multi_mode == 'late':
                multiview = self.multiview_data[pid][scene_id][:]
            elif CONF.fuse_multi_mode == 'early':
                multiview = self.multiview_data[pid][scene_id]
                point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        # sub-sampling points, choices is the selected points inds
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        ema_point_cloud, ema_choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        #point_cloud, choices = random_sampling(point_cloud, 5000, return_choices=True)
        #ema_point_cloud, ema_choices = random_sampling(point_cloud, 8000, return_choices=True)

        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        #ema_pcl_color = ema_pcl_color[ema_choices]
        pcl_color = pcl_color[choices]

        if self.use_multiview and CONF.fuse_multi_mode == 'late':
            multiview = multiview[choices]

        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((CONF.max_num_obj, 6)) # max_num_obj is 128
        target_bboxes_mask = np.zeros((CONF.max_num_obj))    
        angle_classes = np.zeros((CONF.max_num_obj,))
        angle_residuals = np.zeros((CONF.max_num_obj,))
        size_classes = np.zeros((CONF.max_num_obj,))
        size_residuals = np.zeros((CONF.max_num_obj, 3))
        size_gts = np.zeros((CONF.max_num_obj, 3))
        ref_box_label = np.zeros(CONF.max_num_obj) # bbox label for reference target
        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target
        point_instance_label = np.zeros(self.num_points) - 1
        point_obj_mask = np.zeros(self.num_points)
        point_object_id = np.zeros(self.num_points) - 1     # the point of what object
        point_ref_mask = np.zeros(self.num_points)
        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < CONF.max_num_obj else CONF.max_num_obj
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:CONF.max_num_obj,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            flip_x_axis = 0
            flip_y_axis = 0
            rot_mat = np.identity(3)
            scale_ratio = np.ones((1, 3))
            if self.augment:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    flip_x_axis = 1
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    flip_y_axis = 1
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

                # Augment point cloud scale: 0.85x-1.15x
                scale_ratio = np.random.random() * 0.3 + 0.85
                scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
                point_cloud[:, 0:3] *= scale_ratio
                target_bboxes[:, 0:3] *= scale_ratio
                target_bboxes[:, 3:6] *= scale_ratio
                if self.use_height:
                    point_cloud[:, -1] *= scale_ratio[0, 0]

            gt_centers = target_bboxes[:, 0:3]
            gt_centers[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number
            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            point_context_mask = np.zeros(self.num_points) - 1
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0)) 
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
                    ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1)) # instance_bboxes label index
                    point_instance_label[ind] = ilabel  # instance_bboxes' index
                    point_object_id[ind] = instance_bboxes[ilabel, -1]
                    point_obj_mask[ind] = 1.0
                    if CONF.use_context_label:  # False, by default
                        if DC.nyu40id2class[int(instance_bboxes[ilabel, -2])] in context_label:
                            point_context_mask[ind] = 1
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]
            size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]
            #----------------#
            # Reference Data #
            #----------------#
            # object id point wise binary label
            if CONF.use_context_label:
                point_ref_mask[point_context_mask > 0] = 0.5
            point_ref_mask[point_object_id == object_id] = 1
            # construct the reference target label for each bbox
            ref_box_label = np.zeros(CONF.max_num_obj)
            for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
                if gt_id == object_id:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical 
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((CONF.max_num_obj))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features [N,3]
        data_dict["ema_point_clouds"] = ema_point_cloud.astype(np.float32) # point cloud data for Teacher

        if self.use_multiview and CONF.fuse_multi_mode == 'late':
            data_dict["multiview"] = multiview.astype(np.float32)
        if self.lang_emb_type == 'glove':
            data_dict["lang_feat"] = lang_feat.astype(np.float32)  # language feature vectors
        elif self.lang_emb_type == 'clip':
            data_dict["lang_feat"] = lang_feat  # language tokens
        else:
            raise NotImplementedError('Language embedding type unsupported!')
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_mask"] = lang_mask.astype(np.float32)
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (CONF.max_num_obj, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (CONF.max_num_obj,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (CONF.max_num_obj,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (CONF.max_num_obj,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (CONF.max_num_obj, 3)
        data_dict['size_gts'] = size_gts.astype(np.float32)  # (CONF.max_num_obj, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (CONF.max_num_obj,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (CONF.max_num_obj) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        #data_dict["ema_pcl_color"] = ema_pcl_color
        data_dict['point_instance_label'] = point_instance_label.astype(np.int64) #, -1 if points aren't in instance 
        data_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
        data_dict['point_object_id'] = point_object_id.astype(np.int64)
        #------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------- points property according to referrence-----------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------
        data_dict['point_ref_mask'] = point_ref_mask.astype(np.int64) # 1, if points are in the target instance according to referrence
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(np.int64)

        #data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start
        data_dict["scene_id"] = scene_id
        #-------------------------------------------------
        #-----------------Data Augmentation---------------
        #-------------------------------------------------
        data_dict['flip_x_axis'] = np.array(flip_x_axis).astype(np.int64)
        data_dict['flip_y_axis'] =  np.array(flip_y_axis).astype(np.int64)
        data_dict['rot_mat'] =  rot_mat.astype(np.float32)
        data_dict['scale'] = np.array(scale_ratio).astype(np.float32)


        return data_dict
    
    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                if CONF.det_class_label == 'main_with_others':
                    raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17
            # unique = 0, multiple = 1
            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        lang = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}

            # tokenize the description
            tokens = data["token"]
            if self.lang_emb_type == 'glove':
                embeddings = np.zeros((CONF.max_des_len, 300))
                for token_id in range(CONF.max_des_len):
                    if token_id < len(tokens):
                        token = tokens[token_id]
                        if token in glove:
                            embeddings[token_id] = glove[token]
                        else:
                            embeddings[token_id] = glove["unk"]
            elif self.lang_emb_type == 'clip' and CONF.word_erase > 0 and (self.split == 'burn_in' or self.split == 'semi_train'):
                embeddings = tokens
            elif self.lang_emb_type == 'clip':
                embeddings = clip.tokenize(' '.join(tokens), truncate=True).squeeze(0)
            else:
                raise NotImplementedError('Language embedding type unsupported!')
            # store
            lang[scene_id][object_id][ann_id] = embeddings

        return lang


    def _load_data(self):
        print("loading %s data..."%self.split)
        # load language features
        self.lang = self._tranform_des()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox

class ScannetReferenceUnlabeledDataset(Dataset):
   
    def __init__(self, scanrefer, scanrefer_all_scene, 
        split="train", 
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False, 
        lang_emb_type='glove'):

        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.lang_emb_type = lang_emb_type

        # load data
        self._load_data()
        self.multiview_data = {}
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = self.scanrefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        if self.lang_emb_type == 'clip' and CONF.word_erase > 0 and (self.split == 'burn_in' or self.split == 'semi_train'):
            lang_len = len(lang_feat)
            choice_idx = np.random.choice(lang_len, int(float(CONF.word_erase) * lang_len))
            # breakpoint()
            for i in choice_idx:
                lang_feat[i] = ""
            # lang_feat[choice_idx] = 'UNK'
            lang_token = lang_feat
            lang_feat = clip.tokenize(' '.join(lang_feat), truncate=True).squeeze(0)
            lang_len = lang_feat.argmax().item() + 1
        if self.lang_emb_type == 'glove':
            lang_token = self.scanrefer[idx]["token"]
            lang_len = len(self.scanrefer[idx]["token"])
            lang_len = lang_len if lang_len <= CONF.max_des_len else CONF.max_des_len
        elif self.lang_emb_type == 'clip':
            lang_token = self.scanrefer[idx]["token"]
            lang_len = lang_feat.argmax().item() + 1
        else:
            raise NotImplementedError('Language embedding type unsupported!')
        lang_mask = np.zeros((CONF.max_des_len, ))
        lang_mask[:lang_len] = 1.0
        if CONF.use_context_label:
            context_label = set()
            for word in lang_token:
                if word in DC.type2class.keys() and word != 'others':
                    context_label.add(DC.type2class[word])

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]     # (50000,9), (xyz,color,normal)
        instance_labels = self.scene_data[scene_id]["instance_labels"] # (50000)
        semantic_labels = self.scene_data[scene_id]["semantic_labels"] # (50000)
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"] # scene_id=562, (21,8)

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            if CONF.fuse_multi_mode == 'late':
                multiview = self.multiview_data[pid][scene_id][:]
            elif CONF.fuse_multi_mode == 'early':
                multiview = self.multiview_data[pid][scene_id]
                point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        # sub-sampling points, choices is the selected points inds
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        ema_point_cloud = random_sampling(point_cloud, self.num_points, return_choices=False)


        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        if self.use_multiview and CONF.fuse_multi_mode == 'late':
            multiview = multiview[choices]

        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((CONF.max_num_obj, 6)) # max_num_obj is 128
        target_bboxes_mask = np.zeros((CONF.max_num_obj))    
        angle_classes = np.zeros((CONF.max_num_obj,))
        angle_residuals = np.zeros((CONF.max_num_obj,))
        size_classes = np.zeros((CONF.max_num_obj,))
        size_residuals = np.zeros((CONF.max_num_obj, 3))
        size_gts = np.zeros((CONF.max_num_obj, 3))
        ref_box_label = np.zeros(CONF.max_num_obj) # bbox label for reference target
        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target
        point_instance_label = np.zeros(self.num_points) - 1
        point_obj_mask = np.zeros(self.num_points)
        point_object_id = np.zeros(self.num_points) - 1     # the point of what object
        point_ref_mask = np.zeros(self.num_points)
        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < CONF.max_num_obj else CONF.max_num_obj
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:CONF.max_num_obj,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            flip_x_axis = 0
            flip_y_axis = 0
            rot_mat = np.identity(3)
            scale_ratio = np.ones((1, 3))
            if self.augment:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    flip_x_axis = 1
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    flip_y_axis = 1
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

                # Augment point cloud scale: 0.85x-1.15x
                scale_ratio = np.random.random() * 0.3 + 0.85
                scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
                point_cloud[:, 0:3] *= scale_ratio
                target_bboxes[:, 0:3] *= scale_ratio
                target_bboxes[:, 3:6] *= scale_ratio
                if self.use_height:
                    point_cloud[:, -1] *= scale_ratio[0, 0]

            gt_centers = target_bboxes[:, 0:3]
            gt_centers[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number
            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            point_context_mask = np.zeros(self.num_points) - 1
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0)) 
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
                    ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1)) # instance_bboxes label index
                    point_instance_label[ind] = ilabel  # instance_bboxes' index
                    point_object_id[ind] = instance_bboxes[ilabel, -1]
                    point_obj_mask[ind] = 1.0
                    if CONF.use_context_label:  # False, by default
                        if DC.nyu40id2class[int(instance_bboxes[ilabel, -2])] in context_label:
                            point_context_mask[ind] = 1
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]
            size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]
            #----------------#
            # Reference Data #
            #----------------#
 
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical 
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((CONF.max_num_obj))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features [N,3]
        data_dict["ema_point_clouds"] = ema_point_cloud.astype(np.float32) # point cloud data for Teacher

        if self.use_multiview and CONF.fuse_multi_mode == 'late':
            data_dict["multiview"] = multiview.astype(np.float32)
        if self.lang_emb_type == 'glove':
            data_dict["lang_feat"] = lang_feat.astype(np.float32)  # language feature vectors
        elif self.lang_emb_type == 'clip':
            data_dict["lang_feat"] = lang_feat  # language tokens
        else:
            raise NotImplementedError('Language embedding type unsupported!')
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_mask"] = lang_mask.astype(np.float32)
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (CONF.max_num_obj, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (CONF.max_num_obj,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (CONF.max_num_obj,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (CONF.max_num_obj,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (CONF.max_num_obj, 3)
        data_dict['size_gts'] = size_gts.astype(np.float32)  # (CONF.max_num_obj, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (CONF.max_num_obj,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (CONF.max_num_obj) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict['point_instance_label'] = point_instance_label.astype(np.int64) #, -1 if points aren't in instance 
        data_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
        data_dict['point_object_id'] = point_object_id.astype(np.int64)
        #------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------- points property according to referrence-----------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------
        

        data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start
        data_dict["scene_id"] = scene_id
        #-------------------------------------------------
        #-----------------Data Augmentation---------------
        #-------------------------------------------------
        data_dict['flip_x_axis'] = np.array(flip_x_axis).astype(np.int64)
        data_dict['flip_y_axis'] =  np.array(flip_y_axis).astype(np.int64)
        data_dict['rot_mat'] =  rot_mat.astype(np.float32)
        data_dict['scale'] = np.array(scale_ratio).astype(np.float32)


        return data_dict
    
    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                if CONF.det_class_label == 'main_with_others':
                    raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        lang = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}

            # tokenize the description
            tokens = data["token"]
            if self.lang_emb_type == 'glove':
                embeddings = np.zeros((CONF.max_des_len, 300))
                for token_id in range(CONF.max_des_len):
                    if token_id < len(tokens):
                        token = tokens[token_id]
                        if token in glove:
                            embeddings[token_id] = glove[token]
                        else:
                            embeddings[token_id] = glove["unk"]
            elif self.lang_emb_type == 'clip' and CONF.word_erase > 0 and (self.split == 'burn_in' or self.split == 'semi_train'):
                embeddings = tokens
            elif self.lang_emb_type == 'clip':
                embeddings = clip.tokenize(' '.join(tokens), truncate=True).squeeze(0)
            else:
                raise NotImplementedError('Language embedding type unsupported!')
            # store
            lang[scene_id][object_id][ann_id] = embeddings

        return lang


    def _load_data(self):
        print("loading %s data..."%self.split)
        # load language features
        self.lang = self._tranform_des()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
    
class ScannetMulti3DReferDataset(Dataset):
       
    def __init__(self, m3drefer, m3drefer_all_scene, 
        split="train", 
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False, 
        lang_emb_type='glove'):

        self.m3drefer = m3drefer
        self.m3drefer_all_scene = m3drefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.lang_emb_type = lang_emb_type

        # load data
        self._load_data()
        self.multiview_data = {}
       
    def __len__(self):
        return len(self.m3drefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.m3drefer[idx]["scene_id"]
        object_id = int(self.m3drefer[idx]["object_id"])
        object_name = " ".join(self.m3drefer[idx]["object_name"].split("_"))
        ann_id = self.m3drefer[idx]["ann_id"]
        
        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        if self.lang_emb_type == 'clip' and CONF.word_erase > 0 and (self.split == 'burn_in'):
            lang_len = len(lang_feat)
            choice_idx = np.random.choice(lang_len, int(float(CONF.word_erase) * lang_len))
            # breakpoint()
            for i in choice_idx:
                lang_feat[i] = ""
            # lang_feat[choice_idx] = 'UNK'
            lang_token = lang_feat
            lang_feat = clip.tokenize(' '.join(lang_feat), truncate=True).squeeze(0)
            lang_len = lang_feat.argmax().item() + 1
        if self.lang_emb_type == 'glove':
            lang_token = self.m3drefer[idx]["token"]
            lang_len = len(self.m3drefer[idx]["token"])
            lang_len = lang_len if lang_len <= CONF.max_des_len else CONF.max_des_len
        elif self.lang_emb_type == 'clip':
            #lang_token = self.m3drefer[idx]["token"]
            lang_len = lang_feat.argmax().item() + 1
        else:
            raise NotImplementedError('Language embedding type unsupported!')
        lang_mask = np.zeros((CONF.max_des_len, ))
        lang_mask[:lang_len] = 1.0
        if CONF.use_context_label:
            context_label = set()
            for word in lang_token:
                if word in DC.type2class.keys() and word != 'others':
                    context_label.add(DC.type2class[word])

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]     # (50000,9), (xyz,color,normal)
        instance_labels = self.scene_data[scene_id]["instance_labels"] # (50000)
        semantic_labels = self.scene_data[scene_id]["semantic_labels"] # (50000)
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"] # scene_id=562, (21,8)

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            if CONF.fuse_multi_mode == 'late':
                multiview = self.multiview_data[pid][scene_id][:]
            elif CONF.fuse_multi_mode == 'early':
                multiview = self.multiview_data[pid][scene_id]
                point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        # sub-sampling points, choices is the selected points inds
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        ema_point_cloud = random_sampling(point_cloud, self.num_points, return_choices=False)

        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]
        if self.use_multiview and CONF.fuse_multi_mode == 'late':
            multiview = multiview[choices]

        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((CONF.max_num_obj, 6)) # max_num_obj is 128
        target_bboxes_mask = np.zeros((CONF.max_num_obj))    
        angle_classes = np.zeros((CONF.max_num_obj,))
        angle_residuals = np.zeros((CONF.max_num_obj,))
        size_classes = np.zeros((CONF.max_num_obj,))
        size_residuals = np.zeros((CONF.max_num_obj, 3))
        size_gts = np.zeros((CONF.max_num_obj, 3))
        ref_box_label = np.zeros(CONF.max_num_obj) # bbox label for reference target
        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target
        point_instance_label = np.zeros(self.num_points) - 1
        point_obj_mask = np.zeros(self.num_points)
        point_object_id = np.zeros(self.num_points) - 1     # the point of what object
        point_ref_mask = np.zeros(self.num_points)
        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < CONF.max_num_obj else CONF.max_num_obj
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:CONF.max_num_obj,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            flip_x_axis = 0
            flip_y_axis = 0
            rot_mat = np.identity(3)
            scale_ratio = np.ones((1, 3))
            if self.augment:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    flip_x_axis = 1
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    flip_y_axis = 1
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

                # Augment point cloud scale: 0.85x-1.15x
                scale_ratio = np.random.random() * 0.3 + 0.85
                scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
                point_cloud[:, 0:3] *= scale_ratio
                target_bboxes[:, 0:3] *= scale_ratio
                target_bboxes[:, 3:6] *= scale_ratio
                if self.use_height:
                    point_cloud[:, -1] *= scale_ratio[0, 0]

            gt_centers = target_bboxes[:, 0:3]
            gt_centers[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number
            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            point_context_mask = np.zeros(self.num_points) - 1
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0)) 
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
                    ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1)) # instance_bboxes label index
                    point_instance_label[ind] = ilabel  # instance_bboxes' index
                    point_object_id[ind] = instance_bboxes[ilabel, -1]
                    point_obj_mask[ind] = 1.0
                    if CONF.use_context_label:  # False, by default
                        if DC.nyu40id2class[int(instance_bboxes[ilabel, -2])] in context_label:
                            point_context_mask[ind] = 1
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]
            size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]
            #----------------#
            # Reference Data #
            #----------------#
            # object id point wise binary label
            if CONF.use_context_label:
                point_ref_mask[point_context_mask > 0] = 0.5
            point_ref_mask[point_object_id == object_id] = 1
            # construct the reference target label for each bbox
            ref_box_label = np.zeros(CONF.max_num_obj)
            for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
                if gt_id == object_id:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical 
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((CONF.max_num_obj))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features [N,3]
        data_dict["ema_point_clouds"] = ema_point_cloud.astype(np.float32) # point cloud data for Teacher

        if self.use_multiview and CONF.fuse_multi_mode == 'late':
            data_dict["multiview"] = multiview.astype(np.float32)
        if self.lang_emb_type == 'glove':
            data_dict["lang_feat"] = lang_feat.astype(np.float32)  # language feature vectors
        elif self.lang_emb_type == 'clip':
            data_dict["lang_feat"] = lang_feat  # language tokens
        else:
            raise NotImplementedError('Language embedding type unsupported!')
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_mask"] = lang_mask.astype(np.float32)
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (CONF.max_num_obj, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (CONF.max_num_obj,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (CONF.max_num_obj,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (CONF.max_num_obj,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (CONF.max_num_obj, 3)
        data_dict['size_gts'] = size_gts.astype(np.float32)  # (CONF.max_num_obj, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (CONF.max_num_obj,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (CONF.max_num_obj) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict['point_instance_label'] = point_instance_label.astype(np.int64) #, -1 if points aren't in instance 
        data_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
        data_dict['point_object_id'] = point_object_id.astype(np.int64)
        #------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------- points property according to referrence-----------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------
        data_dict['point_ref_mask'] = point_ref_mask.astype(np.int64) # 1, if points are in the target instance according to referrence
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(np.int64)

        #data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start
        data_dict["scene_id"] = scene_id
        #-------------------------------------------------
        #-----------------Data Augmentation---------------
        #-------------------------------------------------
        data_dict['flip_x_axis'] = np.array(flip_x_axis).astype(np.int64)
        data_dict['flip_y_axis'] =  np.array(flip_y_axis).astype(np.int64)
        data_dict['rot_mat'] =  rot_mat.astype(np.float32)
        data_dict['scale'] = np.array(scale_ratio).astype(np.float32)


        return data_dict
    
    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                if CONF.det_class_label == 'main_with_others':
                    raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.m3drefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.m3drefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        lang = {}
        for data in self.m3drefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}

            # tokenize the description
            tokens = data["token"]
            if self.lang_emb_type == 'glove':
                embeddings = np.zeros((CONF.max_des_len, 300))
                for token_id in range(CONF.max_des_len):
                    if token_id < len(tokens):
                        token = tokens[token_id]
                        if token in glove:
                            embeddings[token_id] = glove[token]
                        else:
                            embeddings[token_id] = glove["unk"]
            elif self.lang_emb_type == 'clip' and CONF.word_erase > 0 and (self.split == 'burn_in'):
                embeddings = tokens
            elif self.lang_emb_type == 'clip':
                embeddings = clip.tokenize(' '.join(tokens), truncate=True).squeeze(0)
            else:
                raise NotImplementedError('Language embedding type unsupported!')
            # store
            lang[scene_id][object_id][ann_id] = embeddings

        return lang


    def _load_data(self):
        print("loading %s data..."%self.split)
        # load language features
        self.lang = self._tranform_des()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.m3drefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
    
class ScannetNr3DDataset(Dataset):
       
    def __init__(self, nr3d, nr3d_all_scene, 
        split="train", 
        num_points=40000,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False, 
        lang_emb_type='glove'):

        self.scanrefer = nr3d
        self.scanrefer_all_scene = nr3d_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.lang_emb_type = lang_emb_type

        # load data
        self._load_data()
        self.multiview_data = {}
       
    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = self.scanrefer[idx]["object_name"] # modified object_name without '_'
        ann_id = self.scanrefer[idx]["ann_id"]

        # get language features
        lang_feat = self.lang[scene_id][str(object_id)][ann_id]
        if self.lang_emb_type == 'clip' and CONF.word_erase > 0 and (self.split == 'burn_in' or self.split == 'semi_train'):
            lang_len = len(lang_feat)
            choice_idx = np.random.choice(lang_len, int(float(CONF.word_erase) * lang_len))
            # breakpoint()
            for i in choice_idx:
                lang_feat[i] = ""
            # lang_feat[choice_idx] = 'UNK'
            lang_token = lang_feat
            lang_feat = clip.tokenize(' '.join(lang_feat), truncate=True).squeeze(0)
            lang_len = lang_feat.argmax().item() + 1
        if self.lang_emb_type == 'glove':
            lang_token = self.scanrefer[idx]["token"]
            lang_len = len(self.scanrefer[idx]["token"])
            lang_len = lang_len if lang_len <= CONF.max_des_len else CONF.max_des_len
        elif self.lang_emb_type == 'clip':
            lang_token = self.scanrefer[idx]["token"]
            lang_len = lang_feat.argmax().item() + 1
        else:
            raise NotImplementedError('Language embedding type unsupported!')
        lang_mask = np.zeros((CONF.max_des_len, ))
        lang_mask[:lang_len] = 1.0
        if CONF.use_context_label:
            context_label = set()
            for word in lang_token:
                if word in DC.type2class.keys() and word != 'others':
                    context_label.add(DC.type2class[word])

        # get pc
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]     # (50000,9), (xyz,color,normal)
        instance_labels = self.scene_data[scene_id]["instance_labels"] # (50000)
        semantic_labels = self.scene_data[scene_id]["semantic_labels"] # (50000)
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"] # scene_id=562, (21,8)

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            ema_point_cloud = copy.deepcopy(point_cloud)
            pcl_color = mesh_vertices[:,3:6]
            ema_pcl_color = copy.deepcopy(pcl_color)
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
            ema_point_cloud = copy.deepcopy(point_cloud)
            ema_pcl_color = copy.deepcopy(pcl_color)
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)
            ema_point_cloud = np.concatenate([ema_point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            if CONF.fuse_multi_mode == 'late':
                multiview = self.multiview_data[pid][scene_id][:]
            elif CONF.fuse_multi_mode == 'early':
                multiview = self.multiview_data[pid][scene_id]
                point_cloud = np.concatenate([point_cloud, multiview],1)
                ema_point_cloud = np.concatenate([ema_point_cloud, multiview],1)


        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        # sub-sampling points, choices is the selected points inds
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        ema_point_cloud, ema_choices = random_sampling(ema_point_cloud, self.num_points, return_choices=True)
        #point_cloud, choices = random_sampling(point_cloud, 5000, return_choices=True)
        #ema_point_cloud, ema_choices = random_sampling(point_cloud, 8000, return_choices=True)

        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        ema_pcl_color = ema_pcl_color[ema_choices]
        pcl_color = pcl_color[choices]

        if self.use_multiview and CONF.fuse_multi_mode == 'late':
            multiview = multiview[choices]

        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((CONF.max_num_obj, 6)) # max_num_obj is 128
        target_bboxes_mask = np.zeros((CONF.max_num_obj))    
        angle_classes = np.zeros((CONF.max_num_obj,))
        angle_residuals = np.zeros((CONF.max_num_obj,))
        size_classes = np.zeros((CONF.max_num_obj,))
        size_residuals = np.zeros((CONF.max_num_obj, 3))
        size_gts = np.zeros((CONF.max_num_obj, 3))
        ref_box_label = np.zeros(CONF.max_num_obj) # bbox label for reference target
        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target
        point_instance_label = np.zeros(self.num_points) - 1
        point_obj_mask = np.zeros(self.num_points)
        point_object_id = np.zeros(self.num_points) - 1     # the point of what object
        point_ref_mask = np.zeros(self.num_points)
        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < CONF.max_num_obj else CONF.max_num_obj
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:CONF.max_num_obj,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            flip_x_axis = 0
            flip_y_axis = 0
            rot_mat = np.identity(3)
            scale_ratio = np.ones((1, 3))
            if self.augment:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    flip_x_axis = 1
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    flip_y_axis = 1
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

                # Augment point cloud scale: 0.85x-1.15x
                scale_ratio = np.random.random() * 0.3 + 0.85
                scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
                point_cloud[:, 0:3] *= scale_ratio
                target_bboxes[:, 0:3] *= scale_ratio
                target_bboxes[:, 3:6] *= scale_ratio
                if self.use_height:
                    point_cloud[:, -1] *= scale_ratio[0, 0]

            gt_centers = target_bboxes[:, 0:3]
            gt_centers[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number
            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered 
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            point_context_mask = np.zeros(self.num_points) - 1
            for i_instance in np.unique(instance_labels):            
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0)) 
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
                    ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1)) # instance_bboxes label index
                    point_instance_label[ind] = ilabel  # instance_bboxes' index
                    point_object_id[ind] = instance_bboxes[ilabel, -1]
                    point_obj_mask[ind] = 1.0
                    if CONF.use_context_label:  # False, by default
                        if DC.nyu40id2class[int(instance_bboxes[ilabel, -2])] in context_label:
                            point_context_mask[ind] = 1
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
            
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]
            size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]
            #----------------#
            # Reference Data #
            #----------------#
            # object id point wise binary label
            if CONF.use_context_label:
                point_ref_mask[point_context_mask > 0] = 0.5
            point_ref_mask[point_object_id == object_id] = 1
            # construct the reference target label for each bbox
            ref_box_label = np.zeros(CONF.max_num_obj)
            for i, gt_id in enumerate(instance_bboxes[:num_bbox,-1]):
                if gt_id == object_id:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]
        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical 
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((CONF.max_num_obj))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features [N,3]
        data_dict["ema_point_clouds"] = ema_point_cloud.astype(np.float32) # point cloud data for Teacher

        if self.use_multiview and CONF.fuse_multi_mode == 'late':
            data_dict["multiview"] = multiview.astype(np.float32)
        if self.lang_emb_type == 'glove':
            data_dict["lang_feat"] = lang_feat.astype(np.float32)  # language feature vectors
        elif self.lang_emb_type == 'clip':
            data_dict["lang_feat"] = lang_feat  # language tokens
        else:
            raise NotImplementedError('Language embedding type unsupported!')
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["lang_mask"] = lang_mask.astype(np.float32)
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (CONF.max_num_obj, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (CONF.max_num_obj,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (CONF.max_num_obj,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (CONF.max_num_obj,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (CONF.max_num_obj, 3)
        data_dict['size_gts'] = size_gts.astype(np.float32)  # (CONF.max_num_obj, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (CONF.max_num_obj,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (CONF.max_num_obj) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["ema_pcl_color"] = ema_pcl_color
        data_dict['point_instance_label'] = point_instance_label.astype(np.int64) #, -1 if points aren't in instance 
        data_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
        data_dict['point_object_id'] = point_object_id.astype(np.int64)
        #------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------- points property according to referrence-----------------------------------------------
        #------------------------------------------------------------------------------------------------------------------------------
        data_dict['point_ref_mask'] = point_ref_mask.astype(np.int64) # 1, if points are in the target instance according to referrence
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(np.int64)

        #data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start
        data_dict["scene_id"] = scene_id
        #-------------------------------------------------
        #-----------------Data Augmentation---------------
        #-------------------------------------------------
        data_dict['flip_x_axis'] = np.array(flip_x_axis).astype(np.int64)
        data_dict['flip_y_axis'] =  np.array(flip_y_axis).astype(np.int64)
        data_dict['rot_mat'] =  rot_mat.astype(np.float32)
        data_dict['scale'] = np.array(scale_ratio).astype(np.float32)


        return data_dict
    
    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                if CONF.det_class_label == 'main_with_others':
                    raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17
            # unique = 0, multiple = 1
            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        lang = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}

            # tokenize the description
            tokens = data["token"]
            if self.lang_emb_type == 'glove':
                embeddings = np.zeros((CONF.max_des_len, 300))
                for token_id in range(CONF.max_des_len):
                    if token_id < len(tokens):
                        token = tokens[token_id]
                        if token in glove:
                            embeddings[token_id] = glove[token]
                        else:
                            embeddings[token_id] = glove["unk"]
            elif self.lang_emb_type == 'clip' and CONF.word_erase > 0 and (self.split == 'burn_in' or self.split == 'semi_train'):
                embeddings = tokens
            elif self.lang_emb_type == 'clip':
                embeddings = clip.tokenize(' '.join(tokens), truncate=True).squeeze(0)
            else:
                raise NotImplementedError('Language embedding type unsupported!')
            # store
            lang[scene_id][object_id][ann_id] = embeddings

        return lang


    def _load_data(self):
        print("loading %s data..."%self.split)
        # load language features
        self.lang = self._tranform_des()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # load scene data
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            # self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_vert.npy")
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
            # self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
