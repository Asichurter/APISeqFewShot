import torch as t
from torch.nn.utils.rnn import pad_sequence


#########################################
# 使用pad_sequence来将序列补齐从而批次化。会同
# 时返回长度信息以便于将pad_sequence还原。
#########################################
def getBatchSequenceFunc(d_type='long'):

    def batchSequences(data):
        seqs = [x[0] for x in data]
        labels = t.LongTensor([x[1] for x in data])

        seqs.sort(key=lambda x: len(x), reverse=True)  # 按长度降序排列
        seq_len = [len(q) for q in seqs]
        seqs = pad_sequence(seqs, batch_first=True)

        if d_type=='long':
            return seqs.long(), labels, seq_len
        elif d_type=='float':
            return seqs.float(), labels, seq_len
        else:
            raise ValueError('无效的数据类型: %s'%d_type)

    return batchSequences


def extractTaskStructFromInput(support, query,
                               unsqueezed=False,
                               is_matrix=False):
        # support_dim_size = 6 if is_embedded else 3
        # query_dim_size = 5 if is_embedded else 2
        # len_dim_base = 1 if not is_embedded else 2      # 对于矩阵序列的输入，序列长度的维度是2([qk, in_channel=1, seq_len, height, width])

        support_dim_size = 3 + 2*is_matrix + unsqueezed
        query_dim_size = 2 + 2*is_matrix + unsqueezed
        len_dim_base = 1 if not unsqueezed else 2

        assert len(support.size()) == support_dim_size, '支持集结构 %s 不符合要求！'%(str(support.size()))
        assert len(query.size()) == query_dim_size, '查询集结构 %s 不符合要求！'%(str(query.size()))

        n = support.size(0)
        k = support.size(1)
        sup_seq_len = support.size(len_dim_base+1)

        qk = query.size(0)
        que_seq_len = query.size(len_dim_base)

        assert sup_seq_len==que_seq_len, \
            '支持集序列长度%d和查询集序列%d长度不同！'%(sup_seq_len,que_seq_len,)

        return n, k, qk, sup_seq_len, que_seq_len

def repeatProtoToCompShape(proto, qk, n):
    proto = proto.repeat((qk, 1, 1)).view(qk, n, -1)

    return proto


def repeatQueryToCompShape(query, qk, n):
    query = query.repeat(n,1,1).transpose(0,1).contiguous().view(qk,n,-1)

    return query

def squEucDistance(v1, v2, neg=False):
    assert v1.size()==v2.size() and len(v1.size())==2, \
        '两组向量形状必须相同，且均为(batch, dim)结构！'

    factor = -1 if neg else 1

    return ((v1-v2)**2).sum(dim=1) * factor

def cosDistance(v1, v2, neg=False, factor=10):
    assert v1.size()==v2.size() and len(v1.size())==2, \
        '两组向量形状必须相同，且均为(batch, dim)结构！'

    factor = -1*factor if neg else factor

    return t.cosine_similarity(v1, v2, dim=1) * factor
    # return ((v1-v2)**2).sum(dim=1) * factor

def protoDisAdapter(support, query, qk, n, dim, dis_type='euc'):
    support = support.view(qk*n, dim)
    query = query.view(qk*n, dim)

    if dis_type == 'euc':
        sim = squEucDistance(support, query, neg=True)
    elif dis_type == 'cos':
        sim = cosDistance(support, query, neg=False)

    return sim.view(qk, n)

