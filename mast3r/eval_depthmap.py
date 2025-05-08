import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized
import pdb
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from mast3r.model import AsymmetricMASt3R, AsymmetricMASt3R_DINOv2, AsymmetricMASt3R_warp, AsymmetricMASt3R_only_warp
from dust3r.model import AsymmetricCroCo3DStereo, AsymmetricCroCo3DStereo_DINOv2,AsymmetricCroCo3DStereo_DINOv2_rope, AsymmetricCroCo3DStereo_ResNet, AsymmetricCroCo3DStereo_cnn, inf  # noqa: F401, needed when loading the model
from mast3r.datasets import get_data_loader  # noqa
from dust3r.losses import ConfLoss, L21  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch, loss_of_one_batch_warp, loss_of_one_batch_only_warp , make_batch_symmetric # noqa
from mast3r.losses import *
import tqdm
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa
import torch.nn.functional as F



def get_args_parser():
    parser = argparse.ArgumentParser('MASt3R evaluating', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricMASt3R",
                        type=str, help="string containing the model to build")
    parser.add_argument("--weights", type=str, help="path to the model weights", default='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric')
    parser.add_argument('--test_criterion', default="Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)", type=str, help="test criterion")
    # dataset#_ScaleShiftInv
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=100, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--disable_cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    return parser

@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for _, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=False,
                                       use_amp=bool(args.amp), ret='loss')
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results

def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader



def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model).from_pretrained(args.weights).to(device)
    model.eval()
    print('Building test dataset {:s}'.format(args.test_dataset))
    data_loader_test = {
        dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
        for dataset in args.test_dataset.split('+')
    }
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion).to(device)
    all_test_stats = {}
    for name, dataloader in data_loader_test.items():
        print(f"\nTesting on {name}...")
        stats = test_one_epoch(model, test_criterion, dataloader, device,
                               epoch=0, log_writer=None, args=args, prefix=name)
        all_test_stats[name] = stats

        print(f"{name} stats:", stats)
    
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    evaluate(args)
    