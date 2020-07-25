import torch
import torch.nn.functional as F


def cal_loss(pred, gold, trg_pad_idx, eps, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = eps
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        loss = F.cross_entropy(pred,
                               gold,
                               ignore_index=trg_pad_idx,
                               reduction='mean')

    return loss


def smooth_ce(output_1, output_2, target_1, target_2, eps):
    loss_1 = cal_loss(output_1, target_1, -100, eps)
    loss_2 = cal_loss(output_2, target_2, -100, eps)
    return loss_1 + loss_2
