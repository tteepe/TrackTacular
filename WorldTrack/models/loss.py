import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import basic


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = basic.reduce_masked_mean(loss, valid)
        return loss


class FocalLoss(torch.nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self, use_distance_weight=False):
        super(FocalLoss, self).__init__()
        self.use_distance_weight = use_distance_weight

    def forward(self, pred, gt):
        """ Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
            Arguments:
                pred (batch x c x h x w)
                gt_regr (batch x c x h x w)
        """
        # find pos indices and neg indices
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        distance_weight = torch.ones_like(gt)
        if self.use_distance_weight:
            w, h = gt.shape[-2:]
            xs = torch.linspace(-1, 1, steps=h, device=gt.device)
            ys = torch.linspace(-1, 1, steps=w, device=gt.device)
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            distance_weight = 9 * torch.sin(torch.sqrt(x * x + y * y)) + 1

        # following paper alpha 2, beta 4
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * distance_weight
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds * distance_weight

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = basic.reduce_masked_mean(mse_loss, pos_mask * valid)
    neg_loss = basic.reduce_masked_mean(mse_loss, neg_mask * valid)
    loss = (pos_loss + neg_loss) * 0.5

    return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, rotbin, rotres):
        loss = compute_rot_loss(output, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='mean')


def compute_bin_loss(output, target, mask):
    # mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

