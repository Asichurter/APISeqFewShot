import torch as t
import torch.nn as nn
import numpy as np

from components.reduction.max import StepMaxReduce
from components.sequence.LSTM import BiLstmEncoder, BiLstmCellEncoder
from components.sequence.CNN import CNNEncoder1D
from components.sequence.transformer import MultiHeadAttention, TransformerEncoder
from utils.training import extractTaskStructFromInput


class IMP(nn.Module):

    def __init__(self, pretrained_matrix, embed_size, **modelParams):
        super(IMP, self).__init__()

        self.DataParallel = modelParams['data_parallel'] if 'data_parallel' in modelParams else False

        sigma = 1 if 'init_sigma' not in modelParams else modelParams['init_sigma']
        alpha = 0.1 if 'alpha' not in modelParams else modelParams['alpha']

        self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, padding_idx=0)
        self.EmbedNorm = nn.LayerNorm(embed_size)
        self.EmbedDrop = nn.Dropout(modelParams['dropout'])

        hidden_size = (1+modelParams['bidirectional'])*modelParams['hidden_size']

        # self.Encoder = BiLstmEncoder(input_size=embed_size,
        #                              **modelParams)
        self.Encoder = BiLstmCellEncoder(input_size=embed_size, **modelParams)

        # self.Encoder = TransformerEncoder(embed_size=embed_size, **modelParams)


        self.MiddleEncoder = None#MultiHeadAttention(mhatt_input_size=hidden_size, **modelParams)

        # self.Decoder = StepMaxReduce()
        self.Decoder = CNNEncoder1D([hidden_size, hidden_size])

        self.Sigma = nn.Parameter(t.FloatTensor([sigma]))
        self.ALPHA = alpha
        self.Dim = (1+modelParams['bidirectional'])*modelParams['hidden_size']
        self.NumClusterSteps = 1 if 'cluster_num_step' not in modelParams else modelParams['cluster_num_step']

        self.Clusters = None
        self.ClusterLabels = None

    def _embed(self, x, lens=None):
        x = self.EmbedDrop(self.EmbedNorm(self.Embedding(x)))
        x = self.Encoder(x, lens)
        if self.MiddleEncoder is not None:
            x = self.MiddleEncoder(x, lens)
        x = self.Decoder(x, lens)
        return x

    def _add_cluster(self, nClusters, protos, radii, cluster_type='unlabeled', ex=None):
        """
        Args:
            nClusters: number of clusters
            protos: [B, nClusters, D] cluster protos
            radii: [B, nClusters] cluster radius,
            cluster_type: ['labeled','unlabeled'], the type of cluster we're adding
            ex: the example to add
        Returns:
            updated arguments
        """
        nClusters += 1
        bsize = protos.size()[0]
        dimension = protos.size()[2]

        zero_count = t.zeros(bsize, 1).cuda()

        d_radii = t.ones(bsize, 1).cuda()

        if cluster_type == 'labeled':
            d_radii = d_radii * t.exp(self.Sigma)
        else:
            d_radii = d_radii * t.exp(self.log_sigma_u)

        if ex is None:
            new_proto = self.base_distribution.data.cuda()
        else:
            new_proto = ex.unsqueeze(0).unsqueeze(0).cuda()

        protos = t.cat([protos, new_proto], dim=1)  # 将当前样本设定为新的类簇
        radii = t.cat([radii, d_radii], dim=1)  # 将可学习参数log_sigma_labeled设定为初始半径加入到半径中
        return nClusters, protos, radii
    
    def _compute_protos(self, h, probs):
        """Compute the prototypes
        Args:
            h: [B, N, D] encoded inputs
            probs: [B, N, nClusters] soft assignment
        Returns:
            cluster protos: [B, nClusters, D]
        """

        h = t.unsqueeze(h, 2)       # [B, N, 1, D]
        probs = t.unsqueeze(probs, 3)       # [B, N, nClusters, 1]
        prob_sum = t.sum(probs, 1)  # [B, nClusters, 1]

        #####################################################################
        # 使用z=1来代替sum=0的cluster，防止除0错误
        zero_indices = (prob_sum.view(-1) == 0).nonzero()
        if t.numel(zero_indices) != 0:
            values = t.masked_select(t.ones_like(prob_sum), t.eq(prob_sum, 0.0))
            prob_sum = prob_sum.put_(zero_indices, values)
        #####################################################################
        protos = h*probs    # [B, N, nClusters, D]
        protos = t.sum(protos, 1)/prob_sum
        return protos

    def estimate_lambda(self, tensor_proto, semi_supervised):
        # estimate lambda by mean of shared sigmas
        # 利用proto方差的均值ρ，来估计λ
        rho = tensor_proto[0].var(dim=0)
        rho = rho.mean()

        if semi_supervised:
            sigma = (t.exp(self.log_sigma_l).data[0] + t.exp(self.log_sigma_u).data[0]) / 2.
        else:
            sigma = t.exp(self.Sigma).data[0]  # σ是可学习的参数

        alpha = self.ALPHA
        lamda = -2 * sigma * np.log(alpha) + self.Dim * sigma * np.log(1 + rho.cpu() / sigma.cpu())

        return lamda

    def delete_empty_clusters(self, tensor_proto, prob, radii, targets, eps=1e-3):
        column_sums = t.sum(prob[0], dim=0).data
        good_protos = column_sums > eps
        idxs = t.nonzero(good_protos).squeeze()
        return tensor_proto[:, idxs, :], radii[:, idxs], targets[idxs]

    def loss(self, logits, targets, labels):
        """Loss function to "or" across the prototypes in the class:
        take the loss for the closest prototype in the class and all negatives.（类内最近的样本用于计算损失值）
        inputs:
            logits [B, N, nClusters] of nll probs for each cluster
            targets [B, N] of target clusters
        outputs:
            weighted cross entropy such that we have an "or" function
            across prototypes in the class of each query
        """
        targets = targets.cuda()
        # determine index of closest in-class prototype for each query
        target_logits = t.ones_like(logits.data) * float('-Inf')
        target_logits[targets] = logits.data[targets]  # 只将标签对应的类簇的位置设定为logits数据，其余位置设置为负无穷
        _, best_targets = t.max(target_logits, dim=1)  # 获得最大logit值的类簇的下标
        # mask out everything...
        weights = t.zeros_like(logits.data)
        # ...then include the closest prototype in each class and unlabeled)
        unique_labels = np.unique(labels.cpu().numpy())
        for l in unique_labels:
            class_mask = labels == l  # shape: [batch, sample]
            class_logits = t.ones_like(logits.data) * float('-Inf')  # shape: [batch, sample, cluster] 
            class_logits[class_mask.repeat(logits.size(0), 1)] = logits[class_mask.repeat(logits.size(0), 1)].data.view(-1)  # 只将logits标签为l为的样本的logit填入
            _, best_in_class = t.max(class_logits, dim=1)                               # 对每个类的所有类簇，只选出距离最近的类簇填入，避免对其他类簇进行惩罚
            weights[range(0, targets.size(0)), best_in_class] = 1.  # 同类只取
        loss = weighted_loss(logits, best_targets, weights)
        return loss.mean()

    def forward(self, support, query, sup_len, que_len, support_labels, query_labels,
                if_cache_data=False):

        if self.DataParallel:
            support = support.squeeze(0)
            sup_len = sup_len[0]
            support_labels = support_labels[0]

        n, k, qk, sup_seq_len, que_seq_len = extractTaskStructFromInput(support, query)

        nClusters = n  # 初始类簇的数量等于类数量
        # batch = self._process_batch(sample, super_classes=super_classes)

        nInitialClusters = nClusters

        # run data through network
        support = self._embed(support.view(n*k, sup_seq_len), sup_len)  # 数据嵌入
        query = self._embed(query, que_len)

        # 此处设定batch=1
        # support_labels = t.arange(0, n)[:, None].repeat(1, k).view(1, -1)
        support_labels = support_labels.unsqueeze(0)
        query_labels = query_labels.unsqueeze(0)
        support = support.view(n*k,-1).unsqueeze(0)
        query = query.view(qk,-1).unsqueeze(0)

        # create probabilities for points
        # _, idx = np.unique(batch.y_train.squeeze().data.cpu().numpy(), return_inverse=True)
        prob_support = one_hot(support_labels, nClusters).cuda()  # 将属于类簇的概率初始化为标签的one-hot

        # make initial radii for labeled clusters
        bsize = support.size()[0]
        radii = t.ones(bsize, nClusters).cuda() * t.exp(
            self.Sigma)  # 初始半径由log_sigma_l初始化(该参数可学习)

        cluster_labels = t.arange(0, nClusters).cuda().long()

        # compute initial prototypes from labeled examples
        # 由于初始时，共有类别个类簇，而且类簇的分配系数是one-hot，因此初始类簇就是类中心
        # shape: [batch, cluster, dim]
        protos = self._compute_protos(support, prob_support)

        # estimate lamda
        lamda = self.estimate_lambda(protos.data, False)

        # loop for a given number of clustering steps
        for ii in range(self.NumClusterSteps):
            # protos = protos.data
            # iterate over labeled examples to reassign first
            for i, ex in enumerate(support[0]):
                # 找到样本label对应的cluster的index
                idxs = t.nonzero(support_labels[0, i] == cluster_labels).squeeze(0)            # TODO: 取0？
                # 计算与标签对应的类簇的距离(由于其他不对应的类簇的距离都是正无穷，求min时直接可忽略)
                distances = self._compute_distances(protos[:, idxs, :], ex.data)
                # print(distances.tolist(), lamda.item())
                if t.min(distances) > lamda:
                    nClusters, protos, radii = self._add_cluster(nClusters, protos, radii,
                                                                       cluster_type='labeled', ex=ex.data)
                    cluster_labels = t.cat([cluster_labels, support_labels[0, [i]].data], dim=0)  # 将样本标签设定为类簇标签

            # perform partial reassignment based on newly created labeled clusters
            if nClusters > nInitialClusters:
                support_targets = support_labels.data[0, :, None] == cluster_labels  # 找到每个样本实际对应的类簇（每一行是每个样本对应的类簇bool）
                prob_support = assign_cluster_radii_limited(protos, support, radii,
                                                          support_targets)  # 样本属于每个类簇的概率

            nTrainClusters = nClusters

            # # iterate over unlabeled examples
            # if batch.x_unlabel is not None:
            #     h_unlabel = self._run_forward(batch.x_unlabel)
            #     h_all = t.cat([h_train, h_unlabel], dim=1)
            #     unlabeled_flag = t.LongTensor([-1]).cuda()
            #
            #     for i, ex in enumerate(h_unlabel[0]):
            #         distances = self._compute_distances(protos, ex.data)
            #         if t.min(distances) > lamda:
            #             nClusters, protos, radii = self._add_cluster(nClusters, protos, radii,
            #                                                                cluster_type='unlabeled', ex=ex.data)
            #             cluster_labels = t.cat([cluster_labels, unlabeled_flag], dim=0)
            #
            #     # add new, unlabeled clusters to the total probability
            #     if nClusters > nTrainClusters:
            #         unlabeled_clusters = t.zeros(prob_support.size(0), prob_support.size(1), nClusters - nTrainClusters)
            #         prob_support = t.cat([prob_support, Variable(unlabeled_clusters).cuda()], dim=2)
            #
            #     prob_unlabel = assign_cluster_radii(Variable(protos).cuda(), h_unlabel, radii)
            #     prob_unlabel_nograd = Variable(prob_unlabel.data, requires_grad=False).cuda()
            #     prob_all = t.cat([Variable(prob_support.data, requires_grad=False), prob_unlabel_nograd], dim=1)
            #
            #     protos = self._compute_protos(h_all, prob_all)
            #     protos, radii, cluster_labels = self.delete_empty_clusters(protos, prob_all, radii, cluster_labels)
            # else:
            protos = protos.cuda()
            protos = self._compute_protos(support, prob_support)
            protos, radii, cluster_labels = self.delete_empty_clusters(protos, prob_support, radii, cluster_labels)

        # 计算query的类簇logits
        logits = compute_logits_radii(protos, query, radii).squeeze()

        # convert class targets into indicators for supports in each class
        labels = query_labels#batch.y_test.data
        labels[labels >= nInitialClusters] = -1

        support_targets = labels[0, :, None] == cluster_labels  # 寻找查询集样本的标签对应的类簇
        loss = self.loss(logits, support_targets,
                         cluster_labels)  # support_targets: 查询样本标签对应的类簇指示; suppott_labels: 类簇的标签

        # map support predictions back into classes to check accuracy
        _, support_preds = t.max(logits.data, dim=1)
        y_pred = cluster_labels[support_preds]

        if if_cache_data:
            self.Clusters = protos
            self.ClusterLabels = cluster_labels
            return support, query, (y_pred==query_labels).sum().item()/query.size(1)

        return y_pred, loss

        #
        # acc_val = t.eq(y_pred, labels[0]).float().mean()
        #
        # return loss, {
        #     'loss': loss.data[0],
        #     'acc': acc_val,
        #     'logits': logits.data[0]
        # }


    def _compute_distances(self, protos, example):
        dist = t.sum((example - protos)**2, dim=2)
        return dist


def log_sum_exp(value, weights, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: t.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = t.max(value, dim=dim, keepdim=True)
        value0 = value - m              # 减去最大值，保持数值稳定
        if keepdim is False:
            m = m.squeeze(dim)
        return m + t.log(t.sum(weights*t.exp(value0),       # 使用给定的weights对logit掩码过滤
                                       dim=dim, keepdim=keepdim))
def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = t.arange(0, num_classes)\
                                            .long()\
                                            .repeat(batch_size, 1)\
                                            .cuda(device)\
                                            .eq(target.data.repeat(num_classes, 1).t())
    else:
        one_hot_mask = t.arange(0, num_classes)\
                                               .long()\
                                               .repeat(batch_size, 1)\
                                               .eq(target.data.repeat(num_classes, 1).t()) # 只选出最大logit的类簇对应位置的logit值
    return logits.masked_select(one_hot_mask)

def weighted_loss(logits, targets, weights):
	logsumexp = log_sum_exp(logits, weights, dim=1, keepdim=False)
	loss_by_class = -1*class_select(logits,targets) + logsumexp             # targets: 最大logit值对应的cluster的下标
	return loss_by_class


def one_hot(indices, depth, dim=-1, cumulative=True):
    """One-hot encoding along dim"""
    new_size = []
    for ii in range(len(indices.size())):
        if ii == dim:
            new_size.append(depth)
        new_size.append(indices.size()[ii])
    if dim == -1:
        new_size.append(depth)

    out = t.zeros(new_size)
    indices = t.unsqueeze(indices, dim)
    out = out.scatter_(dim, indices.data.type(t.LongTensor), 1.0)

    return out

def assign_cluster_radii_limited(cluster_centers, data, radii, target_labels):
    """Assigns data to cluster center, using K-Means.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        radii: [B, K] Cluster radii.
    Returns:
        prob: [B, N, K] Soft assignment.
    """
    target_labels = target_labels.unsqueeze(0)      # 设置batch=1
    logits = compute_logits_radii(cluster_centers, data, radii) # [B, N, K]
    class_logits = (t.min(logits).data-100)*t.ones(logits.data.size()).cuda()
    class_logits[target_labels] = logits.data[target_labels]        # 只将每个样本对应类簇的logits置为负平方欧式距离，其余置为一个非常小的值(-100-min)
    logits_shape = logits.size()        # shape: [batch, data, cluster]
    bsize = logits_shape[0]
    ndata = logits_shape[1]
    ncluster = logits_shape[2]
    prob = t.softmax(class_logits, dim=-1)        # 每个样本根据logits对每个cluster进行softmax归一获得assignment以便可微
    return prob

def compute_logits_radii(cluster_centers, data, radii, prior_weight=1.):
    """Computes the logits of being in one cluster, squared Euclidean.

    Args:
        cluster_centers: [B, K, D] Cluster center representation.
        data: [B, N, D] Data representation.
        radii: [B, K] Cluster radii.
    Returns:
        log_prob: [B, N, K] logits.
    """
    cluster_centers = t.unsqueeze(cluster_centers, 1)   # [B, 1, K, D]
    data = t.unsqueeze(data, 2)  # [B, N, 1, D]
    dim = data.size()[-1]
    radii = t.unsqueeze(radii, 1)  # [B, 1, K]  K=类簇数量          
    neg_dist = -t.sum((data - cluster_centers)**2, dim=3)   # [B, N, K]         # 每个样本到每个类簇的欧式距离平方

    logits = neg_dist / 2.0 / (radii)                               # ((x-μ)^2)/(2σ)
    norm_constant = 0.5*dim*(t.log(radii) + np.log(2*np.pi))            

    logits = logits - norm_constant
    return logits