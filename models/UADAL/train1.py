# UADAL training and evaluation on pre-extracted features.

from __future__ import print_function
from data_loader.get_loader import get_dataset_information, get_loader
import random
from utils import utils as utils
from models.basenet1 import *
import datetime
import numpy as np
import time
import datetime
import argparse
import warnings
from data_loader.dataset import UDADataset
import os
import sys

warnings.filterwarnings("ignore")


def main(args):
    t1 = time.time()
    sum_str = ''
    args_list = [str(arg) for arg in vars(args)]
    args_list.sort()
    for arg in args_list:
        sum_str += '{:>20} : {:<20} \n'.format(arg, getattr(args, arg))
    utils.setGPU(args.set_gpu)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    args.device = torch.device("cuda:%s" % (0))

    torch.set_num_threads(1)
    source_data = os.path.join(args.data_dir, args.source_domain)
    target_data = os.path.join(args.data_dir, args.target_domain)
    evaluation_data = os.path.join(args.data_dir, args.target_domain)
    num_class = args.num_class
    # source_data, target_data, evaluation_data, num_class = get_dataset_information(args.dataset, args.source_domain, args.target_domain)
    train_transforms, test_transforms = utils.bring_data_transformation()

    src_dset = UDADataset(
        source_data,
        source_data,
        num_class,
        train_transforms,
        test_transforms,
        is_target=False,
        batch_size=args.batch_size,
    )
    src_train_dset, _, _ = src_dset.get_dsets()
    target_dset = UDADataset(
        target_data,
        target_data,
        num_class,
        train_transforms,
        test_transforms,
        is_target=True,
        batch_size=args.batch_size,
    )
    target_dset.get_dsets()

    dataset_info = '%s_%s' % (args.source_domain, args.target_domain)

    save_info = '%s#%s#%s#%s#%s' % (
        args.exp_code,
        args.model,
        args.net,
        dataset_info,
        args.seed,
    )

    result_model_dir = utils.get_save_dir(args, save_info)

    logger = utils.bring_logger(result_model_dir)
    args.logger = logger
    args.logger.info(sum_str)
    args.logger.info('=' * 30)

    if args.model == 'UADAL':
        from models.model_UADAL import UADAL

        model = UADAL(args, num_class, src_dset, target_dset)
    elif args.model == 'cUADAL':
        from models.model_cUADAL import cUADAL

        model = cUADAL(args, num_class, src_dset, target_dset)

    model.train_init()
    model.test(0)
    model.build_model()
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch code for UADAL/cUADAL')

    # Data Level
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--source_domain', type=str, default='C', help='E, M, Me ')
    parser.add_argument('--target_domain', type=str, default='C', help='E, M, Me ')
    parser.add_argument(
        '--seed', type=int, default=0, metavar='S', help='random seed (default: 0)'
    )
    parser.add_argument(
        '--num_class', type=int, default=4, metavar='S', help='random seed (default: 0)'
    )

    ## Model Level
    parser.add_argument('--model', type=str, default='UADAL', help='UADAL, cUADAL')
    parser.add_argument(
        '--net',
        type=str,
        default='resnet50',
        metavar='B',
        help='resnet50, efficientnet, densenet, vgg',
    )
    parser.add_argument(
        '--bottle_neck_dim',
        type=int,
        default=256,
        metavar='B',
        help='bottle_neck_dim for the classifier network.',
    )
    parser.add_argument(
        '--bottle_neck_dim2',
        type=int,
        default=500,
        metavar='B',
        help='bottle_neck_dim for the classifier network.',
    )

    ## Iteration Level
    parser.add_argument(
        '--warmup_iter',
        type=int,
        default=2000,
        metavar='S',
        help='warmup iteration for posterior inference',
    )
    parser.add_argument(
        '--training_iter', type=int, default=100, metavar='S', help='training_iter'
    )
    parser.add_argument(
        '--update_term',
        type=int,
        default=10,
        metavar='S',
        help='update term for posterior inference',
    )

    ## Loss Level
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        metavar='fixmatch',
        help='threshold for fixmatch',
    )
    parser.add_argument(
        '--ls_eps',
        type=float,
        default=0.1,
        metavar='LR',
        help='label smoothing for classification',
    )

    ## Optimization Level
    parser.add_argument(
        '--update_freq_D',
        type=int,
        default=1,
        metavar='S',
        help='freq for D in optimization.',
    )
    parser.add_argument(
        '--update_freq_G',
        type=int,
        default=1,
        metavar='S',
        help='freq for G in optimization.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        metavar='N',
        help='input batch size for training (default: 32)',
    )
    parser.add_argument(
        '--scheduler', type=str, default='cos', help='learning rate scheduler'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='label smoothing for classification',
    )
    parser.add_argument(
        '--e_lr',
        type=float,
        default=0.002,
        metavar='LR',
        help='label smoothing for classification',
    )
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.1,
        metavar='LR',
        help='label smoothing for classification',
    )

    parser.add_argument(
        '--opt_clip',
        type=float,
        default=0.1,
        metavar='LR',
        help='label smoothing for classification',
    )
    ## etc:
    parser.add_argument(
        '--exp_code',
        type=str,
        default='Test',
        metavar='S',
        help='random seed (default: 0)',
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default='results_resnet',
        metavar='S',
        help='random seed (default: 0)',
    )
    parser.add_argument('--set_gpu', type=int, default=0, help='gpu setting 0 or 1')
    parser.add_argument(
        '--no_cuda', action='store_true', default=False, help='disable cuda'
    )
    args = parser.parse_args()
    main(args)
