import torch as t
import torch.nn as nn

##########################################
# 一种简单的快速获取序列数据嵌入的方式，即取序列上的平均值
##########################################
class FastTextEncoder(nn.Module):

    def __init__(self, pretrained, embed_size, dropout, **kwargs):
        super(FastTextEncoder, self).__init__()

        self.Embedding = nn.Embedding.from_pretrained(pretrained,
                                                      padding_idx=0)
        self.LayerNorm = nn.LayerNorm(embed_size)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x, lens=None):
        # x shape: [batch, seq, dim]
        # average over sequence dim
        batch_size, seq_len = x.size()
        x = self.Embedding(x)
        x = self.LayerNorm(x)
        x = x.sum(dim=1)


        if lens is not None:
            lens_expand = t.FloatTensor(lens).unsqueeze(1).expand_as(x)
        else:
            lens_expand = t.FloatTensor([seq_len]*batch_size).unsqueeze(1).expand_as(x)

        lens_expand = lens_expand.cuda()
        x = x / lens_expand

        return self.Dropout(x)
