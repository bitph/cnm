import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer
import pdb

class MainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = 0.1
        self.target_stride = config['target_stride']
        self.vocab_size = config['vocab_size']
        self.gauss_w = config["gauss_w"]
        self.neg = config['neg']
        self.feat_pool_size = config['feat_pool_size']
        self.num_props = config['num_props']
        self.max_width = config['max_width'] if 'max_width' in config else 1
        # if config['target_stride'] > 1:
        #     self.avg_pool = nn.AvgPool1d(config['target_stride'], config['target_stride'])
        # else:
        #     self.avg_pool = None
        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        
        self.trans = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)
        self.fc_gauss = nn.Linear(config['hidden_size'], self.num_props*2)
        # self.gauss_center = nn.Linear(config['hidden_size'], self.num_props)
        # self.gauss_width = nn.Linear(config['hidden_size'], self.num_props)
        # self.fc_gauss = nn.Sequential(
        #     nn.Linear(config['hidden_size'], config['hidden_size']),
        #     nn.ReLU(),
        #     nn.Linear(config['hidden_size'], len(config['prop_width'])*2)
        # )
        # for param in self.fc_gauss.parameters():
            # torch.nn.init.constant_(param, 0)
        # torch.nn.init.constant_(self.fc_gauss.weight, 0)
        # torch.nn.init.constant_(self.fc_gauss.bias, 0)
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)
        self.unfrozen()
    
    def comp_parameters(self):
        # return list(self.parameters())
        params = []
        for name, param in self.named_parameters():
            if 'gauss' not in name:
                params.append(param)
        return params
    
    def gauss_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if 'comp' not in name:
                params.append(param)
        return params

    def unfrozen(self):
        for name, param in self.named_parameters():
            if 'gauss' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def frozen(self):
        for name, param in self.named_parameters():
            if 'gauss' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        # pdb.set_trace()
        bsz, n_frames, _ = frames_feat.shape
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        ori_frames_feat = frames_feat
        # if self.avg_pool is not None:
        #     frames_feat = self.avg_pool(frames_feat.transpose(-1, -2)).transpose(-1, -2)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        # proposals scoring
        # pdb.set_trace()
        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        gauss_param = torch.sigmoid(self.fc_gauss(h[:, -1])).view(bsz*self.num_props, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1] * self.max_width
        if not self.training:
            return {
                'width': gauss_width,
                'center': gauss_center,
            } 

        # gauss_center = torch.sigmoid(self.gauss_center(h[:, -1])).view(bsz)
        # gauss_width = torch.exp(self.gauss_width(h[:, -1])).view(bsz)

        props_chosen = torch.tensor([0, n_frames], dtype=torch.long).view(1, 1, 2).expand(bsz, self.num_props, 2).cuda()
        props_feat, props_len, props_mask = self._generate_proposals_feat(ori_frames_feat, props_chosen, clip_size=4)
        
        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)
        # if self.avg_pool is not None:
        #     props_feat = self.avg_pool(props_feat.transpose(-1, -2)).transpose(-1, -2)
        
        # semantic completion
        words_feat1, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat1 = words_feat1 + words_pos
        words_feat1 = words_feat1[:, :-1]
        words_id1 = words_id
        words_mask1 = words_mask[:, :-1]

        words_mask1 = words_mask1.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_id1 = words_id1.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_feat1 = words_feat1.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, words_mask1.size(1), -1)

        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=gauss_weight, need_weight=True)
        words_logit = self.fc_comp(h)

        if torch.isnan(words_logit).any():
            pdb.set_trace()

        if self.neg:
            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=1-gauss_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)
            # neg_words_logit_2 = None

            # _, neg_h = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=1-gauss_weight)
            props_feat = props_feat.view(bsz, self.num_props, props_mask.size(1), -1)[:, 0]
            props_mask = props_mask.view(bsz, self.num_props, -1)[:, 0]
            words_feat1 = words_feat1.view(bsz, self.num_props, words_mask1.size(1), -1)[:, 0]
            words_mask1 = words_mask1.view(bsz, self.num_props, -1)[:, 0]
            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2)
            neg_words_logit_1 = self.fc_comp(neg_h_1)
        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None

        weights = None
        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'words_logit': words_logit,  # [nb * nw, seq_len, vocab]
            'words_id': words_id,  # [nb * nw, seq_len]
            'weights': weights,
            'words_mask': words_mask[:, :-1],
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'attn_weight': attn_weight,
        }
    

    def generate_gauss_weight(self, props_len, center, width):
        # pdb.set_trace()
        weight = []
        for l in props_len:
            pos = torch.linspace(0, 1, l).cuda()
            weight.append(pos)
        max_len = props_len.max()
        for i in range(len(weight)):
            weight[i] = F.pad(weight[i], [0, max_len - props_len[i]])
        weight = torch.stack(weight, 0)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(0.1)
        weight = torch.exp(-self.gauss_w*(weight - center).pow(2)/width.pow(2))
        # weight = torch.exp(-width * (weight - center).pow(2))
        return weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            # num_masked_words = max(2 * l // 3, 1) 
            # num_masked_words = max(l, 1) 
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            # print(p)
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        # exit(0)
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words
    
    def _mask_vis_feats(self, vis_feats, vis_len):
        token = self.vis_mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.frame_fc(token)

        masked_feat = []
        for i, l in enumerate(vis_len):
            l = int(l)
            num_masked_feats = max(l // 3, 1)
            masked_feat.append(torch.zeros([vis_feats.size(1)]).byte().cuda())
            mask_len = min(num_masked_feats, 3)
            num_mask = (num_masked_feats-1) // mask_len + 1
            choices = []
            for i in range(num_mask):
                st = np.random.randint(i*l//num_mask, (i+1)*l//num_mask-mask_len+1)
                choices += [j for j in range(st, st+mask_len)]
            # choices = np.random.choice(np.arange(0, l), 3, replace=False)
            masked_feat[-1][choices] = 1
        masked_feat = torch.stack(masked_feat, 0).unsqueeze(-1)
        masked_feat_vec = vis_feats.new_zeros(*vis_feats.size()) + token
        masked_feat_vec = masked_feat_vec.masked_fill_(masked_feat == 0, 0)
        vis_feats1 = vis_feats.masked_fill(masked_feat == 1, 0) + masked_feat_vec
        return vis_feats1, masked_feat

    def _select_proposals(self, props, props_valid, props_align,
                          random_p=0.0, num_proposals=3, tau=0.6):
        bsz = props.size(0)
        props = props.view(bsz, -1, 2)
        props_valid = props_valid.view(bsz, -1)

        props_chosen = []
        props_idx = []

        def choose(size):
            if np.random.rand() < random_p:
                get_id = np.random.choice(np.arange(0, size), replace=False)
            else:
                get_id = 0
            return get_id

        for i, a in enumerate(props_align):
            a = a.contiguous().view(-1).masked_fill(props_valid[i] == 0, 0)

            # reorder
            idx = torch.argsort(a, descending=True)
            props1 = props[i].index_select(dim=0, index=idx)

            # remove illegal
            kidx = props1[:, 0] >= 0
            idx = idx[kidx]
            props1 = props1[kidx]

            pid = choose(props1.size(0))
            cp, cp_idx = [props1[pid]], [idx[pid]]
            for _ in range(1, num_proposals):
                tmp = cp[-1].unsqueeze(0).expand(props1.size(0), 2)
                iou = calculate_IoU_batch((tmp[:, 0].float(), tmp[:, 1].float()),
                                          (props1[:, 0].float(), props1[:, 1].float()))
                kidx = iou < tau
                if int(kidx.sum()) > 2:
                    idx = idx[kidx]
                    props1 = props1[kidx]
                pid = choose(props1.size(0))
                cp.append(props1[pid])
                cp_idx.append(idx[pid])

            cp, cp_idx = torch.stack(cp, 0), torch.stack(cp_idx, 0)
            # print(cp, cp_idx)
            props_chosen.append(cp)
            props_idx.append(cp_idx)
            # exit(0)
        props_chosen = torch.stack(props_chosen, 0)
        props_idx = torch.stack(props_idx, 0)
        # print(props_chosen)
        return props_chosen, props_idx

    def _generate_proposals_feat(self, frames_feat, props, n_clips=16, clip_size=None):
        props_feats = []
        props_len = []

        for f, p in zip(frames_feat, props):
            for s, e in p:
                s, e = int(s) * self.target_stride, int(e) * self.target_stride
                clip_len = e - s
                if clip_size is not None:
                    n_clips = clip_len // clip_size
                idx = np.linspace(start=0, stop=clip_len - 1, num=n_clips).astype(np.int32)
                try:
                    props_feats.append(f[s:e+1][idx])
                except IndexError:
                    print(f.size(), (s, e))
                    exit(0)
                props_len.append(props_feats[-1].size(0))
        # print(props_len)
        # exit(0)
        max_len = max(props_len)
        for i in range(len(props_feats)):
            props_feats[i] = F.pad(props_feats[i], [0, 0, 0, max_len - props_len[i]])

        props_feats = torch.stack(props_feats, 0)
        props_len = torch.from_numpy(np.asarray(props_len).astype(np.int64)).cuda()
        props_mask = _generate_mask(props_feats, props_len)
        return props_feats, props_len, props_mask

    def _generate_neg_proposals_feat(self, frames_feat, props, n_clips=16, clip_size=None):
        props_feats = []
        props_len = []

        for f, p in zip(frames_feat, props):
            for s, e in p:
                s, e = int(s) * self.target_stride, int(e) * self.target_stride
                clip_len = f.size(0) - (e - s)
                if clip_size is not None:
                    n_clips = clip_len // clip_size
                idx = np.linspace(start=0, stop=clip_len - 1, num=n_clips).astype(np.int32)
                try:
                    # props_feats.append(f[s:e+1][idx])
                    props_feats.append(torch.cat([f[:s], f[e:]], dim=0)[idx])
                except IndexError:
                    print(f.size(), (s, e))
                    exit(0)
                props_len.append(props_feats[-1].size(0))
        # print(props_len)
        # exit(0)
        max_len = max(props_len)
        for i in range(len(props_feats)):
            props_feats[i] = F.pad(props_feats[i], [0, 0, 0, max_len - props_len[i]])

        props_feats = torch.stack(props_feats, 0)
        props_len = torch.from_numpy(np.asarray(props_len).astype(np.int64)).cuda()
        props_mask = _generate_mask(props_feats, props_len)
        return props_feats, props_len, props_mask


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
