import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import copy
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter

import povdp.utils as utils

from swin3d.Swin3D.SemanticSeg.model.Swin3D_RGB import Swin3D

from swin3d.Swin3D.SemanticSeg.util.voxelize import voxelize_and_inverse
from swin3d.Swin3D.SemanticSeg.util.common_util import (
    AverageMeter,
    intersectionAndUnionGPU,
    find_free_port,
    poly_learning_rate,
    smooth_loss,
)
from swin3d.Swin3D.data_utils.indoor3d_util import (
    point_label_to_obj, 
    point_cloud_to_bev
)
from swin3d.Swin3D.SemanticSeg.util import transform
from swin3d.Swin3D.SemanticSeg.util.logger import get_logger

from functools import partial
import MinkowskiEngine as ME
from torch.profiler import profile, record_function, ProfilerActivity
import importlib


def worker_init_fn(worker_id, args):
    random.seed(args.manual_seed + worker_id)


def main_process(args):
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )


def segment(
        point_cloud: np.ndarray, 
        model: Swin3D, 
        args
):
    args.train_gpu = args.train_gpu[:1]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    setup_environment(args)
    segmented_points = None

    if args.multiprocessing_distributed:
        raise NotImplementedError(
            "Multiprocessing distributed is not supported"
        )
    else:
        segmented_points = main_worker_sp(
            point_cloud, 
            model, 
            args
        )

    return segmented_points


def setup_environment(args):
    if args.train_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in args.train_gpu
        )
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False


def main_worker_sp(
        point_cloud: np.ndarray, 
        model: Swin3D, 
        args
):

    global logger, writer
    logger = get_logger(args.save_path + "/eval_pts")
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.num_classes))
    # logger.info(model)
    logger.info(
        "# of Model parameters: {}".format(sum([x.nelement() for x in model.parameters()]))
    )

    model = model.cuda()

    assert (".pth" in args.weight) and os.path.isfile(args.weight)

    if args.weight:
        weight_for_innner_model = args.weight_for_innner_model
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            weights = checkpoint["state_dict"]
            if list(weights.keys())[0].startswith("module."):
                logger.info("=> Loading multigpu weights with module. prefix...")
                weights = {
                    k.partition("module.")[2]: weights[k] for k in weights.keys()
                }
            if weight_for_innner_model:
                model.backbone.load_state_dict(weights)
            else:
                model.load_state_dict(weights)
            
            logger.info(
                "=> loaded weight '{}' (epoch {})".format(
                    args.weight, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    vote_num = args.vote_num
    if vote_num > 1:
        point_transform = transform.Compose(
            [
                transform.RandomRotate(along_z=True),
            ]
        )
    else:
        point_transform = None

    coord, feat = point_cloud[:, 0:3], point_cloud[:, 3:6]
    feat = feat / 127.5 - 1
    input_coord = coord.copy()
    input_feat = feat.copy()
    coord, feat, inverse_map = process_point(
        coord,
        feat,
        voxel_size=.04,
        point_transform=point_transform,
    )
    coord, feat, inverse_map = (coord,), (feat,), (inverse_map,)
    offset, count = list(), 0
    inverse_list = list()
    for item, inverse in zip(coord, inverse_map):
        inverse_list.append(inverse + count)
        count += item.shape[0]
        offset.append(count)

    pred_seg_label = segment_points(
        torch.cat(coord), 
        torch.cat(feat), 
        torch.IntTensor(offset), 
        torch.cat(inverse_list),   
        model, 
        args, 
        vote_num
    ).cpu().numpy()
    
    torch.cuda.empty_cache()

    output = np.concatenate(
        [input_coord, input_feat, pred_seg_label[:, None]], 
        axis=-1
    )
    
    # objs = glob.glob("step_point_*.obj")
    # if not objs:
    #     index = 0
    # else:
    #     indices = [int(obj.split('_')[-1].split('.')[0]) for obj in objs]
    #     index = max(indices) + 1
    # point_label_to_obj(
    #     point_cloud=output, 
    #     out_filename=f'step_point_{index}.obj', 
    # )

    return output


def main_worker_mp(
        point_cloud: np.ndarray, 
        model: Swin3D, 
        args
):
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * args.ngpus_per_node + args.gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    global logger, writer
    logger = get_logger(args.save_path + "/eval_pts")
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    # logger.info(model)
    logger.info(
        "# of Model parameters: {}".format(sum([x.nelement() for x in model.parameters()]))
    )

    model = model.cuda()

    assert (".pth" in args.weight) and os.path.isfile(args.weight)

    if args.weight:
        weight_for_innner_model = args.weight_for_innner_model
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            weights = checkpoint["state_dict"]
            if list(weights.keys())[0].startswith("module."):
                logger.info("=> Loading multigpu weights with module. prefix...")
                weights = {
                    k.partition("module.")[2]: weights[k] for k in weights.keys()
                }
            if weight_for_innner_model:
                model.backbone.load_state_dict(weights)
            else:
                model.load_state_dict(weights)
            if main_process():
                logger.info(
                    "=> loaded weight '{}' (epoch {})".format(
                        args.weight, checkpoint["epoch"]
                    )
                )
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    vote_num = args.vote_num
    if vote_num > 1:
        point_transform = transform.Compose(
            [
                transform.RandomRotate(along_z=True),
            ]
        )
    else:
        point_transform = None

    coord, feat = point_cloud[:, 0:3], point_cloud[:, 3:6]
    feat = feat / 127.5 - 1
    input_coord = coord.copy()
    input_feat = feat.copy()
    coord, feat, inverse_map = process_point(
        coord,
        feat,
        voxel_size=.04,
        point_transform=point_transform,
    )
    coord, feat, inverse_map = (coord,), (feat,), (inverse_map,)
    offset, count = list(), 0
    inverse_list = list()
    for item, inverse in zip(coord, inverse_map):
        inverse_list.append(inverse + count)
        count += item.shape[0]
        offset.append(count)

    pred_seg_label = segment_points(
        torch.cat(coord), 
        torch.cat(feat), 
        torch.IntTensor(offset), 
        torch.cat(inverse_list),   
        model, 
        args,
        vote_num
    ).cpu().numpy()
    
    torch.cuda.empty_cache()

    output = np.concatenate(
        [input_coord, input_feat, pred_seg_label[:, None]], 
        axis=-1)


def process_point(
        coord,
        feat,
        voxel_size=0.04,
        point_transform=None,
):
    if point_transform:
        # coord, feat, label = transform(coord, feat, label)
        color = feat[:, 0:3]
        coord, color = point_transform(coord, color)
        feat[:, 0:3] = color

    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = coord.astype(np.float32)
        coord = coord / voxel_size
        int_coord = coord.astype(np.int32)

        unique_map, inverse_map = voxelize_and_inverse(int_coord, voxel_size)
        # print(len(unique_map), len(inverse_map))
        coord = coord[unique_map]
        feat = feat[unique_map]

    # coord_min = np.min(coord, 0)
    # coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    inverse_map = torch.LongTensor(inverse_map)
    return coord, feat, inverse_map


def voted_output(
        coord, 
        feat, 
        offset, 
        inverse_map,  
        model, 
        vote_num, 
        args
    ):
    output_pts_voted = 0

    for it in range(vote_num):    
        print(f"iteration {it + 1}")    
        print(f"coord ({coord.shape})")
        print(f"feat ({feat.shape})")
        print(f"offset ({offset.shape})")
        print(f"inverse map ({inverse_map.shape})")

        if args.yz_shift:
            coord = coord[:, [0, 2, 1]]
            if "normal" in args.data_name:
                print(f"normal")
                feat = feat[:, [0, 1, 2, 3, 5, 4]]

        offset_ = offset.clone()
        print(f"offset_ before ({offset_.shape}): {offset_}")
        offset_[1:] = offset_[1:] - offset_[:-1]
        print(f"offset_ after ({offset_.shape}): {offset_}")
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        output_coord, output_feat, output_offset = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        output_inverse_map = inverse_map.cuda(non_blocking=True)
        print(f"batch shape: {batch.shape}")
        print(f"feat shape: {output_feat.shape}")

        assert batch.shape[0] == feat.shape[0]

        if args.concat_xyz:
            output_feat = torch.cat([output_feat, output_coord], 1)

        with torch.no_grad():
            print(f"Start inferencing...")
            print(f"feat shape: {output_feat.shape}")
            print(f"coord shape: {output_coord.shape}")
            print(f"batch shape: {batch.shape}")
            output = model(output_feat, output_coord, batch)
            output_pts = output[output_inverse_map]
            output_pts = F.normalize(output_pts, p=1, dim=1)

        output_pts_voted += output_pts
    output_pts_voted /= vote_num
    return output_pts_voted


def segment_points(
        coord, 
        feat, 
        offset, 
        inverse_map,  
        model, 
        args, 
        vote_num=12
    ):
    if main_process(args):
        logger.info(">>>>>>>>>>>>>>>> Start Segmentation >>>>>>>>>>>>>>>>")
    batch_time = AverageMeter()

    torch.cuda.empty_cache()

    model.eval()

    if vote_num > 12:
        try:
            save_path = (
                args.weight.split(".")[0]
                + "/"
                + args.val_split
                + f"_vote{vote_num}"
            )
        except AttributeError:
            save_path = (
                args.weight.split(".")[0]
                + "/val"
                + f"_vote{vote_num}"
            )
    else:
        try:
            save_path = args.weight.split(".")[0] + "/" + args.val_split
        except AttributeError:
            save_path = args.weight.split(".")[0] + "/val"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_pts = voted_output(
        coord, 
        feat, 
        offset, 
        inverse_map, 
        model, 
        vote_num, 
        args
    )
    output = output_pts.max(1)[1]

    # batch_time.update(time.time() - end)
    # end = time.time()
    logger.info(
        "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) ".format(
            batch_time=batch_time
        )
    )
    
    logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    return output
