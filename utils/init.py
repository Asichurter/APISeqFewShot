import torch.nn.init as init

def LstmInit(m):
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        for name, par in m.named_parameters():
            if name.find('weight') != -1:
                init.orthogonal_(par)
            elif name.find('bias') != -1:
                init.constant_(par, 0)
    elif classname.find('Linear') != -1:
        for name, par in m.named_parameters():
            if name.find('weight') != -1:
                init.normal_(par, 0, 0.01)
            elif name.find('bias') != -1:
                init.constant_(par, 0)
#