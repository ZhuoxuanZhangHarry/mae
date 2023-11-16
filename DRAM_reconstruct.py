import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn

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
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

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

def get_labels(images, bool_masked_pos, device: torch.device, normlize_target: bool = True, patch_size: int = 16):
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        # calculate the predict label
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
        unnorm_images = images * std + mean  # in [0, 1]

    if normlize_target:
        images_squeeze = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
        images_norm = (images_squeeze - images_squeeze.mean(dim=-2, keepdim=True)
                    ) / (images_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        # we find that the mean is about 0.48 and standard deviation is about 0.08.
        images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
    else:
        images_patch = rearrange(unnorm_images, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    B, _, C = images_patch.shape
    labels = images_patch[bool_masked_pos].reshape(B, -1, C)
    return labels

@torch.no_grad()
def evaluate_DRAM(mae_model, data_loader, device):
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

    # loss lst
    loss_lst = []

    # future TODO: use vector instead of for loop?
    for imgs, target in metric_logger.log_every(data_loader, 10, header):
        imgs = imgs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # get labels TODO: why?
        # labels = get_labels(images, bool_masked_pos, device, normlize_target, patch_size)
        # print("labels: ", labels)
    
        # calculate loss
        with torch.cuda.amp.autocast():
            # TODO: is this mse loss?
            loss, _, _ = mae_model(imgs, mask_ratio=0.75)
        
        loss_value = loss.item()
        
        loss_lst.append(loss_value)

        with torch.cuda.amp.autocast():
            rec_imgs = imgs

            ## TODO: perform patch attack on imgs

            ## TODO: reconstruct imgs
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
    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    # create mae_model
    # mae_model = models_vit.__dict__[args.model](
    #     num_classes=args.nb_classes,
    #     drop_path_rate=args.drop_path,
    #     global_pool=args.global_pool,
    # )
    mae_model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    mae_model.to(device)

    # config parameters
    model_without_ddp = mae_model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        mae_model = torch.nn.parallel.DistributedDataParallel(mae_model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = mae_model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # load model
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)


    # eval
    test_stats = evaluate_DRAM(mae_model, data_loader, device)
    print(f"Accuracy of the network on the {len(dataset)} test images: {test_stats['acc1']:.1f}%")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
