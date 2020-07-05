import torch as t
import torch.nn as nn

from utils.training import getMaskFromLens

class StepMaxReduce(nn.Module):

    def __init__(self):
        super(StepMaxReduce, self).__init__()

    def forward(self, x, lens=None):
        dim = x.size(2)

        if lens is not None:
            mask = getMaskFromLens(lens).unsqueeze(-1).repeat(1,1,dim)
            x.masked_fill_(mask, value=float('-inf'))

        # shape: [batch, step, dim] -> [batch, dim]
        return t.max(x, dim=1).values