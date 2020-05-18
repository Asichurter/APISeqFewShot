import torch as t
import torch.nn as nn

class DeepSet(nn.Module):
    
    def __init__(self, embed_dim,
                 hidden_dim=64,
                 dropout=0.5,
                 **kwargs):
        super(DeepSet, self).__init__()

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

        self.g = nn.Sequential(
            nn.Linear(2*embed_dim, hidden_dim),     # cat of x and supplement
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )


    def forward(self, x, lens=None):
        # sup shape: [size, feature]
        compl = x.repeat((x.size(0), 1, 1))

        compl = self.h(compl).sum(dim=1)

        except_term = self.h(x)

        compl = compl - except_term     # complementary set does not contain the x term

        residual = self.g(t.cat((x, compl), dim=1))

        return residual + x





