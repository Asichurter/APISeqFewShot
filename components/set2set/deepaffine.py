import torch as t
import torch.nn as nn


class DeepAffine(nn.Module):

    def __init__(self, embed_dim,
                 hidden_dim=128,
                 dropout=0.5,
                 **kwargs):
        super(DeepAffine, self).__init__()

        self.h = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.w = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),  # cat of x and supplement
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.b = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),  # cat of x and supplement
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x, lens=None):
        # x shape: [batch, size(seq), feature]
        seq_len, batch_size = x.size(1), x.size(0)

        compl = x.repeat(1, seq_len, 1)

        # compute element-wise set mapping (map and sum)
        # compl shape: [batch, size(seq), feature]
        # here 'size' dim gets same result for set aggregation
        compl = self.h(compl).view(batch_size, seq_len, seq_len, -1).sum(dim=2)

        except_term = self.h(x)

        compl = compl - except_term  # complementary set does not contain itself

        weight = self.w(t.cat((x, compl), dim=2))
        weight = weight + t.ones_like(weight).cuda()

        bias = self.b(t.cat((x, compl), dim=2))

        return weight * x + bias





