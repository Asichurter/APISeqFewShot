import torch as t
import torch.nn as nn

class DeepSet(nn.Module):
    
    def __init__(self, embed_dim,
                 deepset_hidden_dim=None,
                 dropout=0.5,
                 **kwargs):
        super(DeepSet, self).__init__()

        if deepset_hidden_dim is None:
            deepset_hidden_dim = 2 * embed_dim#128#

        self.h = nn.Sequential(
            nn.Linear(embed_dim, deepset_hidden_dim),
            nn.LayerNorm(deepset_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(deepset_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.g = nn.Sequential(
            nn.Linear(2*embed_dim, 2*deepset_hidden_dim),     # cat of x and supplement
            nn.LayerNorm(2*deepset_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2*deepset_hidden_dim, embed_dim),
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

        compl = compl - except_term     # complementary set does not contain itself

        residual = self.g(t.cat((x, compl), dim=2))

        return residual + x





