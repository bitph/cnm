import argparse
import time
import os
from pathlib import Path

from utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, default=None, required=True,
                        help='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--tag', default='base', type=str)

    parser.add_argument('--rank_w', default=None, type=float)
    parser.add_argument('--adv_w', default=None, type=float)
    parser.add_argument('--lambda_', default=None, type=float)
    parser.add_argument('--margin_1', default=None, type=float)
    parser.add_argument('--margin_2', default=None, type=float)
    parser.add_argument('--gauss_w', default=None, type=float)
    parser.add_argument('--div_margin', default=None, type=float)
    parser.add_argument('--props', default=None, type=int)
    parser.add_argument('--gamma', default=None, type=float)
    parser.add_argument('--epoch', default=None, type=int)
    parser.add_argument('--max_width', default=None, type=float)
    parser.add_argument('--vote', action='store_true')
    parser.add_argument('--num_decoder_layer1', default=None, type=int)
    parser.add_argument('--num_decoder_layer2', default=None, type=int)

    return parser.parse_args()


def main(kargs):
    import logging
    import numpy as np
    import random
    import torch
    from runners import MainRunner

    seed = 8
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if kargs.log_dir:
        Path(kargs.log_dir).mkdir(parents=True, exist_ok=True)
        log_filename = time.strftime("%Y-%m-%d_%H-%M-%S.log", time.localtime())
        log_filename = os.path.join(kargs.log_dir, "{}_{}".format(kargs.tag, log_filename))
    else:
        log_filename = None
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    # logging.info('base seed {}'.format(seed))
    args = load_json(kargs.config_path)
    args['train']['model_saved_path'] = os.path.join(args['train']['model_saved_path'], kargs.tag)
    if kargs.rank_w is not None:
        args['loss']['rank_w'] = kargs.rank_w
    if kargs.adv_w is not None:
        args['loss']['adv_w'] = kargs.adv_w
    if kargs.lambda_ is not None:
        args['loss']['lambda'] = kargs.lambda_
    if kargs.margin_1 is not None:
        args['loss']['margin_1'] = kargs.margin_1
    if kargs.margin_2 is not None:
        args['loss']['margin_2'] = kargs.margin_2
    if kargs.gauss_w is not None:
        args['model']['config']['gauss_w'] = kargs.gauss_w
    if kargs.div_margin is not None:
        args['loss']['div_margin'] = kargs.div_margin
    if kargs.props is not None:
        args['model']['config']['num_props'] = kargs.props
        if kargs.props > 5:
            args['train']['batch_size'] = 32
        else:
            args['train']['batch_size'] = 64
    if kargs.gamma is not None:
        args['model']['config']['gamma'] = kargs.gamma
    if kargs.max_width is not None:
        args['model']['config']['max_width'] = kargs.max_width
    if kargs.epoch is not None:
        args['train']['max_num_epochs'] = kargs.epoch
    if kargs.num_decoder_layer1 is not None:
        args['model']['config']['DualTransformer']['num_decoder_layers1'] = kargs.num_decoder_layer1
    if kargs.num_decoder_layer2 is not None:
        args['model']['config']['DualTransformer']['num_decoder_layers2'] = kargs.num_decoder_layer2
    args['vote'] = kargs.vote
    logging.info(str(args))

    runner = MainRunner(args)

    if kargs.resume:
        runner._load_model(kargs.resume)
    if kargs.eval:
        # import pdb
        # pdb.set_trace()
        runner.eval()
        # for epoch in range(1, 51):
            # runner._load_model('/S1/MIPL/zhengmh/SCN/checkpoints/activitynet/cleaned/i3d_w5/model-%d.pt'%epoch)
            # runner.eval()
            # runner.eval(epoch=epoch-1)
            # runner.eval(save='results/Charades/DoG/inter_adv10.1_clamp/model-%d.pkl'%epoch, epoch=epoch-1)
        # runner.eval(save='results/ActivityNet/gauss/clip4clip/model-19.pkl')
        return
    runner.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
