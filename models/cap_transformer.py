"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

# from torchtext import data
# from torchtext import datasets
# from pycrayon import CrayonClient
# from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL

import random
import string
import sys
import math
# import spacy
import uuid
import numpy as np

import pdb
# import contexts

INF = 1e10

def positional_encodings_like(x, t=None):
    if t is None:
        positions = torch.arange(0, x.size(1)).float()
        if x.is_cuda:
           positions = positions.cuda(x.get_device())
    else:
        positions = t
    encodings = torch.zeros(*x.size()[1:])
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())


    for channel in range(x.size(-1)):
        if channel % 2 == 0:
            encodings[:, channel] = torch.sin(
                positions / 10000 ** (channel / x.size(2)))
        else:
            encodings[:, channel] = torch.cos(
                positions / 10000 ** ((channel - 1) / x.size(2)))
    return Variable(encodings)

def mask(targets, out):
    mask = (targets != 0)
    out_mask = mask.unsqueeze(-1).expand_as(out)
    return targets[mask], out[out_mask].view(-1, out.size(-1))

# torch.matmul can't do (4, 3, 2) @ (4, 2) -> (4, 3)
def matmul(x, y):
    if x.dim() == y.dim():
        return x @ y
    if x.dim() == y.dim() - 1:
        return (x.unsqueeze(-2) @ y).squeeze(-2)
    return (x @ y.unsqueeze(-2)).squeeze(-2)

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))

class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal

    def forward(self, query, key, value, q_mask, k_mask):
        dot_products = matmul(query, key.transpose(1, 2))
        if query.dim() == 3 and (self is None or self.causal):
            tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
            if key.is_cuda:
                tri = tri.cuda(key.get_device())
            dot_products.data.sub_(tri.unsqueeze(0))
        attn_weight = F.softmax(dot_products / self.scale, dim=-1)
        if q_mask is not None:
            attn_weight = attn_weight * q_mask.unsqueeze(2)
            attn_weight = attn_weight / attn_weight.sum(dim=-1, keepdim=True)
        if k_mask is not None:
            attn_weight = attn_weight * k_mask.unsqueeze(1)
            attn_weight = attn_weight / attn_weight.sum(dim=-1, keepdim=True)
        return matmul(self.dropout(attn_weight), value)

class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value, q_mask, k_mask):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return self.wo(torch.cat([self.attention(q, k, v, q_mask, k_mask)
                          for q, k, v in zip(query, key, value)], -1))

class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))

class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_hidden, n_heads, drop_ratio, causal=False):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio, causal=causal),
            d_model, drop_ratio)
        self.attention = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x, encoding=None, x_mask=None, encoding_mask=None):
        x = self.selfattn(x, x, x, None, x_mask)
        if encoding is not None:
            x = self.attention(x, encoding, encoding, None, encoding_mask)
        return self.feedforward(x)

class Encoder(nn.Module):
    def __init__(self, d_model, d_hidden, n_vocab, n_layers, n_heads,
                 drop_ratio):
        super().__init__()
        # self.linear = nn.Linear(d_model*2, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x, mask=None):
        # x = self.linear(x)
        x = x+positional_encodings_like(x)
        x = self.dropout(x)
        if mask is not None:
            x = x*mask
        encoding = []
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x*mask
            encoding.append(x)
        return encoding

class VisDecoder(nn.Module):

    def __init__(self, d_model, d_hidden, n_layers, n_heads,
                 drop_ratio):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x, encoding, x_mask, encoding_mask):
        x = x+positional_encodings_like(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoding, x_mask, encoding_mask)
        return x

class LanDecoder(nn.Module):

    def __init__(self, d_model, d_hidden, vocab, n_layers, n_heads,
                 drop_ratio):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_hidden, n_heads, drop_ratio, causal=True)
             for i in range(n_layers)])
        self.out = nn.Linear(d_model, len(vocab))
        self.dropout = nn.Dropout(drop_ratio)
        self.d_model = d_model
        self.vocab = vocab
        self.d_out = len(vocab)

    def forward(self, x, encoding=None, x_mask=None, encoding_mask=None):
        x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
        x = x+positional_encodings_like(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoding, x_mask, encoding_mask)
        return x

    def greedy(self, encoding, T):
        B, _, H = encoding.size()
        # change T to 20, max # of words in a sentence
        # T = 40
        # T *= 2
        prediction = Variable(encoding.data.new(B, T).long().fill_(
            self.vocab.stoi['<pad>']))
        hiddens = [Variable(encoding.data.new(B, T, H).zero_())
                   for l in range(len(self.layers) + 1)]
        embedW = self.out.weight * math.sqrt(self.d_model)
        hiddens[0] = hiddens[0] + positional_encodings_like(hiddens[0])
        for t in range(T):
            if t == 0:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(Variable(
                    encoding.data.new(B).long().fill_(
                        self.vocab.stoi['<init>'])), embedW)
            else:
                hiddens[0][:, t] = hiddens[0][:, t] + F.embedding(prediction[:, t - 1],
                                                                embedW)
            hiddens[0][:, t] = self.dropout(hiddens[0][:, t])
            for l in range(len(self.layers)):
                x = hiddens[l][:, :t + 1]
                x = self.layers[l].selfattn(hiddens[l][:, t], x, x)
                hiddens[l + 1][:, t] = self.layers[l].feedforward(
                    self.layers[l].attention(x, encoding, encoding))

            _, prediction[:, t] = self.out(hiddens[-1][:, t]).max(-1)
        return hiddens, prediction



class Transformer(nn.Module):

    def __init__(self, lan_encoder, vis_encoder):
        super().__init__()
        self.lan_encoder = lan_encoder
        self.vis_decoder = vis_encoder

    def forward(self, x, s, x_mask=None, s_mask=None):
        encoding = self.lan_encoder(s)
        out = self.vis_decoder(x, encoding, x_mask, s_mask)

        return out


class RealTransformer(nn.Module):

    def __init__(self, vis_encoder, lan_encoder):
        super().__init__()
        self.vis_encoder = vis_encoder
        self.lan_encoder = lan_encoder

    def denum(self, data):
        return ' '.join(self.lan_encoder.vocab.itos[i] for i in data).replace(
            ' <eos>', '').replace(' <pad>', '').replace(' .', '').replace('  ', '')

    def forward(self, x, s, x_mask=None, s_mask=None):
        # pdb.set_trace()
        encoding = self.vis_encoder(x, None, x_mask, None)
        h = self.lan_encoder(s[:, :-1].contiguous(), encoding, s_mask[:, :-1].contiguous(), x_mask)
        logits = self.lan_encoder.out(h)

        return logits

    def greedy(self, x, x_mask, T):
        encoding = self.vis_encoder(x, None, x_mask, None)

        _, pred = self.lan_encoder.greedy(encoding, T)
        sent_lst = []
        for i in range(pred.data.size(0)):
            sent_lst.append(self.denum(pred.data[i]))
        return sent_lst


def positional_encodings(x, D):
    # input x a vector of positions
    encodings = torch.zeros(x.size(0), D)
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())
    encodings = Variable(encodings)

    for channel in range(D):
        if channel % 2 == 0:
            encodings[:,channel] = torch.sin(
                x / 10000 ** (channel / D))
        else:
            encodings[:,channel] = torch.cos(
                x / 10000 ** ((channel - 1) / D))
    return encodings


class DropoutTime1D(nn.Module):
    '''
        assumes the first dimension is batch, 
        input in shape B x T x H
        '''
    def __init__(self, p_drop):
        super(DropoutTime1D, self).__init__()
        self.p_drop = p_drop

    def forward(self, x):
        if self.training:
            mask = x.data.new(x.data.size(0),x.data.size(1), 1).uniform_()
            mask = Variable((mask > self.p_drop).float())
            return x * mask
        else:
            return x * (1-self.p_drop)

    def init_params(self):
        pass

    def __repr__(self):
        repstr = self.__class__.__name__ + ' (\n'
        repstr += "{:.2f}".format(self.p_drop)
        repstr += ')'
        return repstr


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


class Caption(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.vocab = vocab
        self.gauss = config['gauss']
        self.frame_emb = nn.Linear(config['frames_input_size'], config['hidden_size']//2)
        self.motion_emb = nn.Linear(config['frames_input_size'], config['hidden_size']//2)
        self.emb_out = nn.Sequential(
            DropoutTime1D(0.1),
            nn.ReLU()
        )

        t_cfg = config['DualTransformer']
        self.vis_encoder = VisDecoder(t_cfg['d_model'], 
                                      d_hidden=t_cfg['d_model']*2,
                                      n_layers=t_cfg['num_decoder_layers1'],
                                      n_heads=t_cfg['num_heads'],
                                      drop_ratio=t_cfg['dropout'])
        self.lan_encoder = LanDecoder(t_cfg['d_model'], 
                                      d_hidden=t_cfg['d_model']*2,
                                      vocab=vocab,
                                      n_layers=t_cfg['num_decoder_layers2'],
                                      n_heads=t_cfg['num_heads'],
                                      drop_ratio=t_cfg['dropout'])
        self.props_model = Transformer(self.lan_encoder, self.vis_encoder)

        self.fc_props = nn.Linear(config['hidden_size'], len(config['prop_width']))
        self.fc_gauss = nn.Linear(config['hidden_size'], len(config['prop_width'])*2)
        torch.nn.init.constant_(self.fc_gauss.weight, 0)
        torch.nn.init.constant_(self.fc_gauss.bias, 0)

        self.cap_model = RealTransformer(self.vis_encoder, self.lan_encoder)

        self.unfrozen()

    def unfrozen(self):
        for name, param in self.named_parameters():
            if 'fc_gauss' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def frozen(self):
        for name, param in self.named_parameters():
            if 'fc_gauss' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, frames_feat, frames_len, words_id, words_len, weights,
                props, props_valid, num_proposals, random_p, tau=0.60, **kwargs):
        # pdb.set_trace()
        B, T, _ = frames_feat.shape

        motion = frames_feat - torch.cat([frames_feat[:, :1], frames_feat[:, :-1]], dim=1)
        x_frame = self.frame_emb(frames_feat.contiguous())
        x_motion = self.motion_emb(motion.contiguous())
        frames_feat = torch.cat([x_frame, x_motion], dim=-1)
        frames_feat = self.emb_out(frames_feat)
        ori_frames_feat = frames_feat
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_id = torch.cat([words_id.new_zeros((B, 1)).fill_(
                        self.vocab.stoi['<init>']), words_id], dim=-1)
        words_mask = _generate_mask(words_id, words_len+1)

        h = self.props_model(frames_feat, words_id, frames_mask, words_mask)
        props_align = torch.sigmoid(self.fc_props(h))  # [nb, nc, np]

        # proposals selection
        props_chosen, props_idx = self._select_proposals(props, props_valid, props_align,
                                                         random_p=random_p, num_proposals=num_proposals,
                                                         tau=tau)
        props_align = props_align.contiguous().view(B, -1)
        props_align = props_align.gather(dim=1, index=props_idx)

        if self.gauss:
            gauss_param = torch.sigmoid(self.fc_gauss(h)).view(B, -1, 2)
            gauss_center = gauss_param[:, :, 0].gather(dim=1, index=props_idx).view(-1)
            gauss_width = gauss_param[:, :, 1].gather(dim=1, index=props_idx).view(-1)
        else:
            gauss_center = None
            gauss_width = None

        props_feat, props_len, props_mask = self._generate_proposals_feat(ori_frames_feat, props_chosen, clip_size=4)
        if self.gauss:
            gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)
        else:
            gauss_weight = None
        
        words_id1, masked_words = self._mask_words(words_id, words_len, weights=weights)
        words_mask1 = words_mask.unsqueeze(1) \
            .expand(B, num_proposals, -1).contiguous().view(B * num_proposals, -1)
        words_id1 = words_id1.unsqueeze(1) \
            .expand(B, num_proposals, -1).contiguous().view(B * num_proposals, -1)
        gt_id = words_id.unsqueeze(1) \
            .expand(B, num_proposals, -1).contiguous().view(B * num_proposals, -1)
        
        if self.gauss:
            props_mask1 = props_mask * gauss_weight
        else:
            props_mask1 = props_mask
        pos_logit = self.cap_model(props_feat, words_id1, props_mask1, words_mask1)
        
        if self.gauss:
            # props_mask1 = props_mask * (1 - gauss_weight)
            props_mask1 = props_mask
            neg_logit = self.cap_model(props_feat, words_id1, props_mask1, words_mask1)
        else:
            neg_logit = None

        return {
            'neg_words_logit': neg_logit,
            'props_chosen': props_chosen,
            'props_idx': props_idx,
            'props_align': props_align,  # [nb, np]
            'words_logit': pos_logit,  # [nb * nw, seq_len, vocab]
            'words_id': gt_id[:, 1:],  # [nb * nw, seq_len]
            'words_mask': words_mask1[:, 1:],
            'weights': None,
            'width': gauss_width,
            'center': gauss_center,
        }

    def forward_neg(self, frames_feat, frames_len, words_id, words_len,
                    weights, props_chosen, props_idx, **kwargs):
        # pdb.set_trace()
        B, T, _ = frames_feat.shape
        num_proposals = props_chosen.size(1)

        motion = frames_feat - torch.cat([frames_feat[:, :1], frames_feat[:, :-1]], dim=1)
        x_frame = self.frame_emb(frames_feat.contiguous())
        x_motion = self.motion_emb(motion.contiguous())
        frames_feat = torch.cat([x_frame, x_motion], dim=-1)
        frames_feat = self.emb_out(frames_feat)
        ori_frames_feat = frames_feat
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_id = torch.cat([words_id.new_zeros((B, 1)).fill_(
                        self.vocab.stoi['<init>']), words_id], dim=-1)
        words_mask = _generate_mask(words_id, words_len+1)

        h = self.props_model(frames_feat, words_id, frames_mask, words_mask)

        if self.gauss:
            gauss_param = torch.sigmoid(self.fc_gauss(h)).view(B, -1, 2)
            gauss_center = gauss_param[:, :, 0].gather(dim=1, index=props_idx).view(-1)
            gauss_width = gauss_param[:, :, 1].gather(dim=1, index=props_idx).view(-1)
        else:
            gauss_center = None
            gauss_width = None

        props_feat, props_len, props_mask = self._generate_proposals_feat(ori_frames_feat, props_chosen, clip_size=4)
        if self.gauss:
            gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)
        else:
            gauss_weight = None
        
        words_id1, masked_words = self._mask_words(words_id, words_len, weights=weights)
        words_mask1 = words_mask.unsqueeze(1) \
            .expand(B, num_proposals, -1).contiguous().view(B * num_proposals, -1)
        words_id1 = words_id1.unsqueeze(1) \
            .expand(B, num_proposals, -1).contiguous().view(B * num_proposals, -1)
        gt_id = words_id.unsqueeze(1) \
            .expand(B, num_proposals, -1).contiguous().view(B * num_proposals, -1)
        
        if self.gauss:
            props_mask1 = props_mask * gauss_weight
        else:
            props_mask1 = props_mask
        pos_logit = self.cap_model(props_feat, words_id1, props_mask1, words_mask1)
        
        if self.gauss:
            # props_mask1 = props_mask * (1 - gauss_weight)
            props_mask1 = props_mask
            neg_logit = self.cap_model(props_feat, words_id1, props_mask1, words_mask1)
        else:
            neg_logit = None

        return {
            'neg_words_logit': neg_logit,
            'words_logit': pos_logit,  # [nb * nw, seq_len, vocab]
            'words_id': gt_id[:, 1:],  # [nb * nw, seq_len]
            'weights': None,
            'words_mask': words_mask1[:, 1:],
            'width': gauss_width,
            'center': gauss_center,
        }

    def _generate_proposals_feat(self, frames_feat, props, n_clips=16, clip_size=None):
        props_feats = []
        props_len = []

        for f, p in zip(frames_feat, props):
            for s, e in p:
                s, e = int(s), int(e) 
                clip_len = e - s
                if clip_size is not None:
                    n_clips = clip_len // clip_size
                idx = np.linspace(start=0, stop=clip_len - 1, num=n_clips).astype(np.int32)
                try:
                    props_feats.append(f[s:e+1][idx])
                except IndexError:
                    # pdb.set_trace()
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
        width = width.unsqueeze(-1).clamp(0.0001)
        weight = torch.exp(-3*(weight - center).pow(2)/width.pow(2))
        return weight

    def _mask_words(self, words_id, words_len, weights=None):
        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros([words_id.size(1)]).byte().cuda())
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            # print(p)
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        # exit(0)
        masked_words = torch.stack(masked_words, 0)
        words_id1 = words_id.masked_fill(masked_words == 1, self.vocab.stoi['<mask>'])
        return words_id1, masked_words
    
    def _mask_vis_feats(self, vis_feats, vis_len):
        token = self.vis_mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.frame_fc(token)

        masked_feat = []
        for i, l in enumerate(vis_len):
            l = int(l)
            num_masked_feats = max(l // 3, 1)
            masked_feat.append(torch.zeros([vis_feats.size(1)]).byte().cuda())
            choices = np.random.choice(np.arange(0, l), num_masked_feats, replace=False)
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

def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou