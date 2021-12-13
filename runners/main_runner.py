import collections
import logging
import os
import time

import numpy as np
import torch

from models.loss import weakly_supervised_loss, cal_nll_loss, margin_ranking_loss
from utils import TimeMeter, AverageMeter

import pickle
import copy
from pathlib import Path
import pdb

def info(msg):
    print(msg)
    logging.info(msg)

class Vocab:
    def __init__(self, keep_vocab):
        super().__init__()
        self.stoi = copy.deepcopy(keep_vocab)
        self.stoi['<pad>'] = 0
        self.stoi['<eos>'] = len(self.stoi)
        self.stoi['<init>'] = len(self.stoi)
        self.stoi['<mask>'] = len(self.stoi)
        self.itos = {v: k for k, v in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

class MainRunner:
    def __init__(self, args):
        self.args = args
        self._build_dataset()

        self.args['model']['config']['prop_width'] = self.args['dataset']['prop_width']
        self.args['model']['config']['target_stride'] = self.args['dataset']['target_stride']
        self.args['model']['config']['vocab_size'] = self.train_set.vocab_size

        self._build_model()
        if 'train' in args:
            self._build_optimizer()
            self.num_updates = 0

    def train(self):
        for epoch in range(1, self.args['train']['max_num_epochs']+1):
            info('Start Epoch {}'.format(epoch))
            self.model_saved_path = self.args['train']['model_saved_path']
            os.makedirs(self.model_saved_path, mode=0o755, exist_ok=True)
            save_path = os.path.join(self.model_saved_path, 'model-{}.pt'.format(epoch))

            self._train_one_epoch(epoch)
            self._save_model(save_path)
            self.eval()
            # self.eval(bias=bias, top_n=5, thresh=0.45)
            info('=' * 60)

    def _train_one_epoch(self, epoch, **kwargs):
        self.model.train()

        def print_log():
            msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, bid, curr_lr)
            for k, v in loss_meter.items():
                msg += '{} = {:.4f}, '.format(k, v.avg)
                v.reset()
            msg += '{:.3f} seconds/batch'.format(1.0 / time_meter.avg)
            info(msg)

        display_n_batches, bid = 50, 0
        time_meter = TimeMeter()
        loss_meter = collections.defaultdict(lambda: AverageMeter())

        for bid, batch in enumerate(self.train_loader, 1):
            # pdb.set_trace()
            self.model.unfrozen()
            self.optimizer.zero_grad()
            net_input = move_to_cuda(batch['net_input'])
            output = self.model(**net_input)
            loss, loss_dict = weakly_supervised_loss(**output, num_props=self.model.num_props, **self.args['loss'])
            # neg_loss, neg_loss_dict = margin_ranking_loss(**output, num_props=self.model.num_props, **self.args['loss'])
            # loss = loss + neg_loss
            # loss_dict.update(neg_loss_dict)
            # backward
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            # update
            self.optimizer.step()

            if self.args['model']['twostep']:
                self.model.frozen()
                self.gauss_optimizer.zero_grad()
                output = self.model(**net_input)
                loss, neg_loss_dict = margin_ranking_loss(**output, num_props=self.model.num_props, **self.args['loss'])
                loss_dict.update(neg_loss_dict)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.gauss_optimizer.step()
            else:
                self.model.frozen()
                self.gauss_optimizer.zero_grad()
                loss, neg_loss_dict = margin_ranking_loss(**output, num_props=self.model.num_props, **self.args['loss'])
                loss_dict.update(neg_loss_dict)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.gauss_optimizer.step()

            self.num_updates += 1
            curr_lr = self.lr_scheduler.step_update(self.num_updates)
            self.gauss_lr_scheduler.step_update(self.num_updates)
            time_meter.update()
            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % display_n_batches == 0:
                print_log()

        if bid % display_n_batches != 0:
            print_log()

    def eval(self, save=None):
        self.model.eval()
        if save is not None:
            save_dict = collections.defaultdict(list)
        with torch.no_grad():
            metrics_logger = collections.defaultdict(lambda: AverageMeter())

            with torch.no_grad():
                for bid, batch in enumerate(self.test_loader, 1):
                    durations = np.asarray([i[1] for i in batch['raw']])
                    gt = np.asarray([i[2] for i in batch['raw']])

                    net_input = move_to_cuda(batch['net_input'])
                    # net_input['props'] = batch['net_input']['props'].expand(len(self.device_ids), -1, -1, -1)
                    # net_input['props_valid'] = batch['net_input']['props_valid'].expand(len(self.device_ids), -1, -1)
                    # forward
                    # pdb.set_trace()
                    time_st = time.time()
                    output = self.model(**net_input)

                    bsz = len(durations)
                    
                    width = output['width'].view(bsz)
                    # torch.clamp(width, max=0.5)
                    center = output['center'].view(bsz)
                    selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
                                                  torch.clamp(center+width/2, max=1)], dim=-1)
                    time_en = time.time()
                    selected_props = selected_props.cpu().numpy()
                    # if top_n > 1:
                    #     num_clips = self.num_clips
                    #     sort_idx = np.argsort(-prob, -1)
                    #     cand_props = list(self.props[sort_idx])  # [bsz, cand_props, 2]
                    #     top_n_selected_props = [selected_props]
                    #
                    #     for it in range(1, top_n):
                    #         ptr_props = top_n_selected_props[-1]
                    #         selected_props = []
                    #         for i in range(bsz):
                    #             p2 = cand_props[i]
                    #             p1 = np.repeat(np.expand_dims(ptr_props[i], 0),
                    #                            p2.shape[0], 0)
                    #
                    #             iou = calculate_IoU_batch2((p1[:, 0], p1[:, 1]), (p2[:, 0], p2[:, 1]))
                    #             keep = iou <= thresh
                    #             # print(keep.shape, cand_props[i].shape)
                    #             cand_props[i] = cand_props[i][keep]
                    #             # print(cand_props[i].shape)
                    #             selected_props.append(cand_props[i][0])
                    #         top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips
                    #         # print(np.asarray(selected_props).shape, selected_props[0].shape)
                    #         top_n_selected_props.append(np.asarray(selected_props))
                    #     top_n_selected_props[-1] = top_n_selected_props[-1] * durations[:, np.newaxis] / num_clips
                    #     res = top_n_metric(top_n_selected_props, gt)
                    # else:
                    # selected_props = selected_props * durations[:, np.newaxis, np.newaxis]
                    gt = gt / durations[:, np.newaxis]
                    res = top_1_metric(selected_props, gt)
                    for key, v in res.items():
                        metrics_logger['R@1,'+key].update(v, bsz)

                    if save is not None:
                        # pdb.set_trace()
                        for key in ['gauss_weight', 'attn_weight']:
                            save_dict[key] += [d for d in output[key].cpu().numpy()]
                        # pred_words_score, pred_words_id = output['words_logit'].max(dim=-1)
                        # save_dict['pred_words_id'] += [d for d in pred_words_id.cpu().numpy()]
                        # save_dict['pred_words_score'] += [d for d in pred_words_score.cpu().numpy()]
                        save_dict['selected_props'] += [d for d in selected_props]
                        # save_dict['ori_props'] += [d for d in ori_props]
                        save_dict['gt'] += [d for d in gt]
                    metrics_logger['Time'].update(time_en-time_st, bsz)

            # for k, v in metrics_logger.items():
                # print('| {} {:.4f}'.format(k, v.avg), end=' ')
            # print('|')
            msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
            info('|'+msg+'|')
            if save is not None:
                Path(save).parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(save_dict, open(save, 'wb'))
            return metrics_logger

    def _build_dataset(self):
        import datasets as da
        import pickle
        from torch.utils.data import DataLoader
        args = self.args['dataset']
        cls = getattr(da, args['dataset'], None)
        # vocab = KeyedVectors.load_word2vec_format(args['vocab_path'], binary=True)
        with open(args['vocab_path'], 'rb') as fp:
            vocab = pickle.load(fp)
        self.train_set = cls(data_path=args['train_data'], vocab=vocab, args=args, is_training=True)
        self.test_set = cls(data_path=args['test_data'], vocab=vocab, args=args)
        self.val_set = cls(data_path=args['val_data'], vocab=vocab, args=args) if args['val_data'] else None
        info('train: {} samples, test: {} samples'.format(len(self.train_set), len(self.test_set)))
        batch_size = self.args['train']['batch_size']

        def worker_init_fn(worker_id):
            def set_seed(seed):
                import random
                import numpy as np
                import torch

                random.seed(seed)
                np.random.seed(seed + 1)
                torch.manual_seed(seed + 3)
                torch.cuda.manual_seed(seed + 4)
                torch.cuda.manual_seed_all(seed + 4)

            set_seed(8 + worker_id)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                       collate_fn=self.train_set.collate_data, num_workers=2,
                                       worker_init_fn=worker_init_fn)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False,
                                      collate_fn=self.test_set.collate_data,
                                      num_workers=1)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False,
                                     collate_fn=self.val_set.collate_data,
                                     num_workers=1) if args['val_data'] else None

    def _build_model(self):
        model_config = self.args['model']
        # print(model_config)
        import models

        device_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        info('GPU: {}'.format(device_ids))
        if model_config['name'] == 'Caption':
            from models.cap_transformer import Caption
            self.model = Caption(model_config['config'], Vocab(self.train_loader.dataset.keep_vocab))
        else:
            self.model = getattr(models, model_config['name'], None)(model_config['config'])
        self.model = self.model.cuda(device_ids[0])
        print(self.model)
        # self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.device_ids = device_ids

    def _build_optimizer(self):
        from optimizers import AdamOptimizer
        from optimizers.lr_schedulers import InverseSquareRootSchedule
        self.model.unfrozen()
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # parameters = self.model.comp_parameters()
        args = self.args['train']["pg"]
        self.optimizer = AdamOptimizer(args, parameters)
        self.lr_scheduler = InverseSquareRootSchedule(args, self.optimizer)

        self.model.frozen()
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # parameters = self.model.gauss_parameters()
        args = self.args['train']["gauss"]
        self.gauss_optimizer = AdamOptimizer(args, parameters)
        self.gauss_lr_scheduler = InverseSquareRootSchedule(args, self.gauss_optimizer)
        

    def _save_model(self, path):
        state_dict = {
            'num_updates': self.num_updates,
            'config': self.args,
            'model_parameters': self.model.state_dict(),
        }
        torch.save(state_dict, path)
        info('save model to {}, num_updates {}.'.format(path, self.num_updates))

    def _load_model(self, path):
        state_dict = torch.load(path)
        self.num_updates = state_dict['num_updates']
        self.lr_scheduler.step_update(self.num_updates)
        parameters = state_dict['model_parameters']
        self.model.load_state_dict(parameters)
        info('load model from {}, num_updates {}.'.format(path, self.num_updates))


def calculate_IoU_batch2(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    # iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


# [nb, 2], [nb, 2]
def top_n_metric(preds, label):
    result = {}
    bsz = preds[0].shape[0]
    top_iou = []
    for pred in preds:
        iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_1_discount_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    discount = (1-np.abs(pred[:, 0]-label[:,0])) * (1-np.abs(pred[:, 1]-label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum((iou >= i / 10).astype(float) * discount) / bsz
    return result

def top_1_metric(pred, label):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU_batch2((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]))
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)
