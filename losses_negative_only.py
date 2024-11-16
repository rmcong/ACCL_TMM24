from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
'''
def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 生成对角单位矩阵
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)  # 512, 1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # 512, 512 构建类标,同类1,否则0
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # augmentation number 2
        # 单独切分每个channel然后按照batch concat起来？变成[batch*channel, HW]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # separate features and concat[batch * contrast_count, HW]
        if self.contrast_mode == 'one':  # 构建单个样本
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':  # 所有样本构建对比学习
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),  # 1024, 1024
            self.temperature)  # 计算每个样本间的contrastive similarity
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # softmax 1024, 1
        logits = anchor_dot_contrast - logits_max.detach()  # normalization to (-1, 0]

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # repeat mask contrast_count times
        n_mask = -(mask-1)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(  # set 0 to diagonal elements and remain 1 for other location
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        # logits_mask = torch.ones_like(mask) - torch.eye(batch_size * anchor_count).to(device)
        mask = mask * logits_mask
        # n_mask = n_mask * logits_mask
        # mask = mask + n_mask


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # loss without calculate loss from same sample
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # n_mean_log_prob_pos = (n_mask * log_prob).sum(1) / n_mask.sum(1)


        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # + nn.BCELoss(reduction='mean')(exp_logits, mask)

        curr_class_mask = torch.zeros_like(labels)  # class_label
        for tc in target_labels:  # current class sample calculate classification loss?
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)  # 512
        loss = curr_class_mask * loss.view(anchor_count, batch_size)  # 2 512

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss

    def forward_shared_and_private(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # if labels is not None and mask is not None:
        #     raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 生成对角单位矩阵
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)  # 512, 1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # 512, 512 构建类标,同类1,否则0
        else:
            mask = mask.contiguous().view(-1, 1).float().to(device)

        contrast_count = features.shape[1]  # augmentation number 2
        # 单独切分每个channel然后按照batch concat起来？变成[batch*channel, HW]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # separate features and concat[batch * contrast_count, HW]
        if self.contrast_mode == 'one':  # 构建单个样本
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':  # 所有样本构建对比学习
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),  # 1024, 1024
            self.temperature)  # 计算每个样本间的contrastive similarity
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # softmax 1024, 1
        logits = anchor_dot_contrast - logits_max.detach()  # normalization to (-1, 0]

        # tile mask
        # mask = mask.repeat(anchor_count)  # repeat mask contrast_count times
        mask = torch.eq(mask, mask.T).float().to(device)
        n_mask = mask[batch_size:,]
        mask = mask[:batch_size,]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(  # set 0 to diagonal elements and remain 1 for other location
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # n_mask = n_mask * logits_mask
        # mask = mask + n_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # loss without calculate loss from same sample
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # negative_mean_log_prob_pos = (n_mask * log_prob).sum(1) / n_mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # + nn.BCELoss(reduction='mean')(exp_logits, mask)
        # curr_class_mask = torch.zeros_like(mask)  # class_label
        # for tc in target_labels:  # current class sample calculate classification loss?
        #     curr_class_mask += (labels == tc)
        # curr_class_mask = curr_class_mask.view(-1).to(device)  # 512
        # loss = loss.view(anchor_count, batch_size)  # 2 512

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss

    def forward_shared_positive(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # if labels is not None and mask is not None:
        #     raise ValueError('Cannot define both `labels` and `mask`')
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # 生成对角单位矩阵
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)  # 512, 1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # 512, 512 构建类标,同类1,否则0
        else:
            mask = mask.contiguous().view(-1, 1).float().to(device)

        contrast_count = features.shape[1]  # augmentation number 2
        # 单独切分每个channel然后按照batch concat起来？变成[batch*channel, HW]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # separate features and concat[batch * contrast_count, HW]
        if self.contrast_mode == 'one':  # 构建单个样本
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':  # 所有样本构建对比学习
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),  # 1024, 1024
            self.temperature)  # 计算每个样本间的contrastive similarity
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # softmax 1024, 1
        logits = anchor_dot_contrast - logits_max.detach()  # normalization to (-1, 0]

        # tile mask
        mask = mask.repeat(anchor_count,1)  # repeat mask contrast_count times
        mask = torch.eq(mask, mask.T).float().to(device)
        n_mask = mask[batch_size:,]
        # mask = mask[:batch_size,]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(  # set 0 to diagonal elements and remain 1 for other location
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # n_mask = n_mask * logits_mask
        # mask = mask + n_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # loss without calculate loss from same sample
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # negative_mean_log_prob_pos = (n_mask * log_prob).sum(1) / n_mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # + nn.BCELoss(reduction='mean')(exp_logits, mask)
        # curr_class_mask = torch.zeros_like(mask)  # class_label
        # for tc in target_labels:  # current class sample calculate classification loss?
        #     curr_class_mask += (labels == tc)
        # curr_class_mask = curr_class_mask.view(-1).to(device)  # 512
        # loss = loss.view(anchor_count, batch_size)  # 2 512

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss