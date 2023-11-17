import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn
import PIL
from timm.data import Mixup
from timm.utils import accuracy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange
import util.misc as misc
import util.lr_sched as lr_sched

import argparse
import datetime
import json
import numpy as np
import os
import time
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import timm
import torchvision.datasets as datasets
# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit
import models_mae

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def load_classifier_model(checkpoint_path, pretrained_model):
    if checkpoint_path:
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            pretrained_model.load_state_dict(checkpoint)
            print("=> load chechpoint found at {}".format(checkpoint_path))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
    return pretrained_model

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

@torch.no_grad()
@torch.cuda.amp.autocast()
def eval_DRAM(classifier_model, ori_imgs, adv_imgs, rec_imgs, target):
    output_ori = classifier_model(ori_imgs)
    pre = torch.max(output_ori.data, 1)
    correct_ori = (pre == target).sum()
    
    output_adv = classifier_model(adv_imgs)
    pre = torch.max(output_adv.data, 1)
    correct_adv = (pre == target).sum()

    output_rec = classifier_model(rec_imgs)
    pre = torch.max(output_rec.data, 1)
    correct_rec = (pre == target).sum()

    return correct_ori, correct_adv, correct_rec


@torch.no_grad()
@torch.cuda.amp.autocast()
def DRAM_reconstruct(mae_model, adv_imgs, mask_ratio=0.75):
    loss, patched_adv_imgs, mask = mae_model(adv_imgs, mask_ratio=0.75)
    # mse loss over patches
    # [batch, patch_num, patch_size**2 * C]
    # TODO:
    adv_imgs = mae_model.unpatchify(patched_adv_imgs)

    return adv_imgs

@torch.no_grad()
@torch.cuda.amp.autocast()
def engine_DRAM(mae_model, data_loader_adv, data_loader_ori, device):
    # set up logger
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # load classifier model
    classifier_model = torchvision.models.resnet50(pretrained=True)
    classifier_model = classifier_model.to(device)

    classifier_model = load_classifier_model("/local/rcs/yunyun/SSDG-main/resnet50.pth", classifier_model)

    # switch to evaluation mode
    classifier_model.eval()
    mae_model.eval()

    # loss func
    MSE_criterion = nn.MSELoss()
    Entropy_loss = nn.CrossEntropyLoss()

    # total correctness
    correct_ori = 0
    correct_adv = 0
    correct_rec = 0

    # iterate ori dataset along with adv
    data_loader_ori_iter = iter(data_loader_ori)

    for adv_imgs, target in metric_logger.log_every(data_loader_adv, 10, header):
        adv_imgs = adv_imgs.to(device, non_blocking=True)
        # [batch, Channel=3, Height, Width]
        target = target.to(device, non_blocking=True)

        ori_imgs = data_loader_ori_iter[0].to(device, non_blocking=True)

        ## TODO: reconstruct imgs to eliminate patch
        rec_imgs = DRAM_reconstruct(mae_model, adv_imgs, mask_ratio=0.75)

        # eval accuracy ori vs adv vs recon
        correct_ori, correct_adv, correct_rec += eval_DRAM(classifier_model, ori_imgs, 
                                                           adv_imgs, rec_imgs, target)
    return


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # load data
    dataset_adv = build_dataset(is_train=False, args=args)
    sampler_adv = torch.utils.data.SequentialSampler(dataset_adv)

    # if args.eval:
    # TODO: load adv set later
    dataset_ori = build_dataset(is_train=False, args=args)
    sampler_ori = torch.utils.data.SequentialSampler(dataset_ori)

    data_loader_adv = torch.utils.data.DataLoader(
        dataset_adv, sampler=sampler_adv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_ori = torch.utils.data.DataLoader(
        dataset_ori, sampler=sampler_ori,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # create mae_model
    # TODO: visualize for debugging purpose
    chkpt_dir = "mae_visualize_vit_large.pth"
    mae_model = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    mae_model.to(device)

    # config parameters
    model_without_ddp = mae_model
    print('Model loaded.')
    print("Model = %s" % str(model_without_ddp))

    # eval
    test_stats = engine_DRAM(mae_model, data_loader_adv, data_loader_ori, device)
    print(f"Accuracy of the network on the {len(dataset_adv)} test images: {test_stats['acc1']:.1f}%")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
