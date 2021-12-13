import torch
import torch.nn.functional as F
import pdb

def cal_nll_loss2(logit, idx, mask, weights=None):
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        # [nb * nw, seq_len]
        nll_loss = (nll_loss * weights).sum(dim=-1)
    # nll_loss = nll_loss.mean()
    return nll_loss.contiguous()


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [nb * nw, seq_len]
    smooth_loss = -logit.sum(dim=-1)  # [nb * nw, seq_len]
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        # [nb * nw, seq_len]
        nll_loss = (nll_loss * weights).sum(dim=-1)
    # nll_loss = nll_loss.mean()
    return nll_loss.contiguous(), mean_acc


def vis_cmp_loss(pred, gt, mask, loss_type):
    bs, n, c = pred.shape
    if loss_type == "mse":
        loss = F.mse_loss(pred, gt, reduction='none')
        if mask is not None:
            loss = loss.masked_fill(mask.view(bs, n, 1)==0, 0)
        return loss.sum(dim=(1, 2)) / mask.sum(dim=-1)
    elif loss_type == "contrastive":
        pred = pred.view(bs*n, c)
        gt = gt.view(bs*n, c).transpose(0, 1)
        score = torch.matmul(pred, gt).view(bs, n, bs*n)
        target = torch.arange(bs*n, dtype=torch.long).view(bs, n).cuda()
        loss, acc = cal_nll_loss(score, target, mask)
        return loss
    else:
        raise NotImplementedError

def weakly_supervised_loss(words_logit, neg_words_logit_1, neg_words_logit_2, words_id, words_mask, num_props, **kwargs):
    # pdb.set_trace()
    bsz = words_logit.size(0) // num_props
    
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    min_nll_loss = nll_loss.view(bsz, num_props).min(dim=-1)[0]
    final_loss = min_nll_loss.mean()
    # final_loss = nll_loss.mean()

    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id, words_mask) 
        final_loss = final_loss + neg_nll_loss_1.mean()
    
    # if neg_words_logit_2 is not None:
    #     neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id, words_mask) 
    #     final_loss = final_loss + neg_nll_loss_2.mean()

        
    loss_dict = {
        'final': final_loss.item(),
        'nll': nll_loss.mean().item(),
        'min_nll': min_nll_loss.mean().item(),
        'acc': acc.item(),
    }
    if neg_words_logit_1 is not None:
        loss_dict.update({
            'neg_nll': neg_nll_loss_1.mean().item(),
            'neg_acc': neg_acc_1.item(),
            })

    return final_loss, loss_dict
    
def margin_ranking_loss(words_logit, neg_words_logit_1, neg_words_logit_2, words_id, words_mask, num_props, **kwargs):
    # pdb.set_trace()
    bsz = words_logit.size(0) // num_props
    
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)
    # min_nll_loss = nll_loss.view(bsz, num_props)

    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id, words_mask)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        global_neg_loss = torch.max(min_nll_loss - neg_nll_loss_1 + kwargs["margin"], tmp_0)
        loss = global_neg_loss.mean()
        # loss = 0.0
    else:
        loss = min_nll_loss.mean()
    
    if neg_words_logit_2 is not None:
        neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id1, words_mask1)
        neg_nll_loss_2 = torch.gather(neg_nll_loss_2.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        inner_neg_loss = torch.max(min_nll_loss - neg_nll_loss_2 + 1.5*kwargs["margin"], tmp_0)
        loss = loss + inner_neg_loss.mean()

    # pdb.set_trace()
    if kwargs["adv"] and num_props > 1:
        # loss = loss - var_loss.mean()
        # inner_neg = nll_loss.view(bsz, num_props).gather(index=idx[:, 1:], dim=-1).min(dim=-1)[0]
        gauss_weight = kwargs['gauss_weight'].view(bsz, num_props, -1)
        # idx = idx.unsqueeze(-1).expand(bsz, num_props, gauss_weight.size(-1))
        # gauss_weight = gauss_weight.gather(index=idx[:, 1:], dim=1)
        gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
        target = torch.diag_embed(gauss_weight.norm(dim=-1).pow(2)).detach()
        # target = torch.eye(num_props).unsqueeze(0).cuda()
        source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
        adv_loss = torch.norm(target - source, dim=(1, 2))**2
        loss = loss + 10.0 * adv_loss.mean()

        # max_nll_loss = nll_loss.view(bsz, num_props).max(dim=-1)[0]
        # tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        # tmp_0.requires_grad = False
        # inner_neg_loss = torch.max(min_nll_loss - max_nll_loss + kwargs["margin"], tmp_0)
        # loss = loss + inner_neg_loss.mean()
    
    idx = torch.argsort(nll_loss.view(bsz, num_props), dim=-1)
    center = kwargs['center'].view(bsz, num_props).gather(index=idx, dim=1)
    width = kwargs['width'].view(bsz, num_props).gather(index=idx, dim=1)
    # center[:, 0] = center[:, 0].detach()
    # var_loss = center.std(dim=-1)
    # loss = loss - var_loss.mean()

    return loss, {
        'nll_1': nll_loss.mean().item(),
        'min_nll_1': min_nll_loss.mean().item(),
        'neg_nll_1': neg_nll_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
        'neg_nll_2': neg_nll_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
        # 'margin': loss.item(),
        'adv_loss': adv_loss.mean().item() if kwargs["adv"] else 0.0,
        # 'var_loss': var_loss.mean().item(),
        'global_margin':  global_neg_loss.mean().item() if neg_words_logit_1 is not None else 0.0,
        'inner_margin': inner_neg_loss.mean().item() if neg_words_logit_2 is not None else 0.0,
        'width': width[:, 0].mean().item(),
        'center': center[:, 0].mean().item(),
    }
