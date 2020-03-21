import torch as t
from torch.nn.utils.rnn import pad_sequence


#########################################
# 使用pad_sequence来将序列补齐从而批次化。会同
# 时返回长度信息以便于将pad_sequence还原。
#########################################
def batchSequences(data):
    seqs = [x[0] for x in data]
    labels = t.LongTensor([x[1] for x in data])

    seqs.sort(key=lambda x: len(x), reverse=True)  # 按长度降序排列
    seq_len = [len(q) for q in seqs]
    seqs = pad_sequence(seqs, batch_first=True)

    return seqs.long(), labels, seq_len


def extractTaskStructFromInput(support, query):
        assert len(support.size()) == 3, '支持集结构不符合 (class, sample, seq) 结构！'
        assert len(query.size()) == 2, '查询集结构不符合 (sample, seq) 结构！'

        n = support.size(0)
        k = support.size(1)
        sup_seq_len = support.size(2)

        qk = query.size(0)
        que_seq_len = query.size(1)

        # assert seq_len==q_seq_len, '支持集中序列长度 %d 与查询集序列长度 %d 不一致！'%(seq_len, q_seq_len)

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
    assert v1.size()==v2.size() and len(v1.size()==2), \
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

