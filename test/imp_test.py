import torch
import numpy as np

def log_sum_exp(value, weights, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m              # 减去最大值，保持数值稳定
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(weights*torch.exp(value0),       # 使用给定的weights对logit掩码过滤
                                       dim=dim, keepdim=keepdim))
def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = target.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.arange(0, num_classes)\
                                            .long()\
                                            .repeat(batch_size, 1)\
                                            .cuda(device)\
                                            .eq(target.data.repeat(num_classes, 1).t())
    else:
        one_hot_mask = torch.arange(0, num_classes)\
                                               .long()\
                                               .repeat(batch_size, 1)\
                                               .eq(target.data.repeat(num_classes, 1).t()) # 只选出最大logit的类簇对应位置的logit值
    return logits.masked_select(one_hot_mask)

def weighted_loss(logits, targets, weights):
	logsumexp = log_sum_exp(logits, weights, dim=1, keepdim=False)
	loss_by_class = -1*class_select(logits,targets) + logsumexp             # targets: 最大logit值对应的cluster的下标
	return loss_by_class

def loss(logits, targets, labels):
    """Loss function to "or" across the prototypes in the class:
    take the loss for the closest prototype in the class and all negatives.（类内最近的样本用于计算损失值）
    inputs:
        logits [B, N, nClusters] of nll probs for each cluster
        targets [B, N] of target clusters
    outputs:
        weighted cross entropy such that we have an "or" function
        across prototypes in the class of each query
    """
    targets = targets
    # determine index of closest in-class prototype for each query
    target_logits = torch.ones_like(logits.data) * float('-Inf')
    target_logits[targets] = logits.data[targets]  # 只将标签对应的类簇的位置设定为logits数据，其余位置设置为负无穷
    _, best_targets = torch.max(target_logits, dim=1)  # 获得最大logit值的类簇的下标
    # mask out everything...
    weights = torch.zeros_like(logits.data)
    # ...then include the closest prototype in each class and unlabeled)
    unique_labels = np.unique(labels.cpu().numpy())
    for l in unique_labels:
        class_mask = labels == l  # shape: [batch, cluster]
        class_logits = torch.ones_like(logits.data) * float('-Inf')  # shape: [batch, sample, cluster]
        class_logits[class_mask.repeat(logits.size(0), 1)] = logits[class_mask].data  # 只将logits标签为l为的样本的logit填入
        _, best_in_class = torch.max(class_logits, dim=1)
        weights[range(0, targets.size(0)), best_in_class] = 1.  # dim=1时，得到与每个类簇最接近的样本的下标
    loss_val = weighted_loss(logits, best_targets, weights)
    return loss_val.mean()

if __name__ == '__main__':
    logits = torch.Tensor([[[-1.5, -2, -1], [-0.5, -1, -2], [-3, -0.5, -4]]])
    q_labels = torch.LongTensor([[0,1,1]])    # 样本标签
    labels = torch.LongTensor([[0,1,1]])      # 类簇标签
    targets = q_labels[:, :, None] == labels   # 样本与类簇的对应

    loss = loss(logits,targets,labels)
