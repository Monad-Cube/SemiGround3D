import os
import sys
import json
import pickle
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.config import CONF
os.environ["CUDA_VISIBLE_DEVICES"] = CONF.gpu   # change the gpu from default.yaml
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval_m3
from models.vgnet import VGNet
from data.scannet.model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()
SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))
M3DREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "multi3drefer_val.json")))
#import debugpy; debugpy.listen(('127.0.0.1', 57000)); debugpy.wait_for_client()

def get_dataloader(args, scanrefer, all_scene_list, split, config):
    # Create Dataset and Dataloader
    val_dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=args.use_height,
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment=False,
        lang_emb_type=args.lang_emb_type
    )
    print("evaluate on {} samples".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
 
    return val_dataset, val_loader

def get_model(args, config):
    # load model
    if args.use_multiview:
        if args.fuse_multi_mode == 'early':
            input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(args.use_height)
        elif args.fuse_multi_mode == 'late':
            input_channels = int(args.use_normal) * 3 + int(args.use_color) * 3 + int(args.use_height)
    else:
        input_channels = int(args.use_normal) * 3 + int(args.use_color) * 3 + int(args.use_height)
    model = VGNet(
        input_feature_dim=input_channels,
        args=CONF,
        data_config=DC,
    ).cuda()

    model_name = "model_last.pth" if args.detection else "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=True)
    model.eval()

    return model

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(args):

    scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
    scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    if args.num_scenes != -1:
        scene_list = scene_list[:args.num_scenes]

    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    return scanrefer, scene_list

def get_m3drefer(args):

    m3drefer = M3DREFER_VAL
    scene_list = sorted(list(set([data["scene_id"] for data in m3drefer])))
    if args.num_scenes != -1:
        scene_list = scene_list[:args.num_scenes]

    m3drefer = [data for data in m3drefer if data["scene_id"] in scene_list]

    return m3drefer, scene_list

def eval_ref(args):
    print("evaluate localization...")
    # constant
    DC = ScannetDatasetConfig()

    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC)

    # model
    model = get_model(args, DC)

    # config
    POST_DICT = {
        "remove_empty_box": True, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.05,
        "dataset_config": DC
    } if not args.no_nms else None

    # random seeds
    seeds = [args.manual_seed] + [2 * i for i in range(args.repeat - 1)]
    #seeds = [args.manual_seed]
    # evaluate
    print("evaluating...")
    score_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "scores.p")
    pred_path = os.path.join(CONF.PATH.OUTPUT, args.folder, "predictions.p")
    gen_flag = (not os.path.exists(score_path)) or args.force or args.repeat > 1
    predictions = {}
    if gen_flag:
        ref_acc_all = []
        ious_all = []
        masks_all = []
        others_all = []
        lang_acc_all = []
        for seed in seeds:
            # reproducibility
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))

            
            for data in tqdm(dataloader):
                for key in data:
                    if key != 'scene_id':
                        data[key] = data[key].cuda()

                # feed
                with torch.no_grad():
                    data = model(data)
                _, data = get_loss(
                    data_dict=data, 
                    config=DC, 
                    args=CONF
                )
                data = get_eval_m3(
                    data_dict=data, 
                    config=DC,
                    reference=True, 
                    use_lang_classifier=not args.no_lang_cls,
                    use_oracle=args.use_oracle,
                    use_cat_rand=args.use_cat_rand,
                    use_best=args.use_best,
                    post_processing=POST_DICT
                )


                # store predictions
                ids = data["scan_idx"].detach().cpu().numpy()
                for i in range(ids.shape[0]):
                    idx = ids[i]
                    scene_id = scanrefer[idx]["scene_id"]
                    object_id = scanrefer[idx]["object_id"]
                    ann_id = scanrefer[idx]["ann_id"]

                    predictions[(scene_id, object_id, ann_id)] = data["pred_bboxes"][i]
    scene_pred = {}
    for key, value in predictions.items():
        scene_id = key[0]
        if key[0] not in scene_pred:
            scene_pred[scene_id] = []

        scene_pred[scene_id].append({
            "object_id": key[1],
            "ann_id": key[2],
            "aabb": value.tolist()
        })
    # TODO, change the path
    prediction_output_root_path = "/workspace/yf/projects/Semi-SPS/scanrefer_pred/10%_data" 
    os.makedirs(prediction_output_root_path, exist_ok=True)
    for scene_id in tqdm(scene_pred.keys(), desc="Saving predictions"):
        with open(os.path.join(prediction_output_root_path, f"{scene_id}.json"), "w") as f:
            json.dump(scene_pred[scene_id], f, indent=2)
    print(f"==> Complete. Saved at: {os.path.abspath(prediction_output_root_path)}")
    print("done!")
    
                
 




if __name__ == "__main__":

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = CONF.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # evaluate
    if CONF.reference: eval_ref(CONF)



