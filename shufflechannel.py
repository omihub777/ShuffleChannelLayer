import sys, os
sys.path.append(os.path.abspath("model"))
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShuffleChannel(nn.Module):
    def __init__(self, p=0.5):
        super(ShuffleChannel, self).__init__()
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p and self.training:
            x = self._shuffle(x)
        return x

    @staticmethod
    def _shuffle(x):
        _,c,_,_ = x.size()
        indices = torch.randperm(c)
        return x[:, indices]

if __name__ == "__main__":
    import torchsummary
    b, c, h, w = 1, 5, 2, 2
    x = torch.arange(c).view(1,c,1,1).expand(b,c,h,w)
    # x[:, 1] = torch.zeros(h,w)
    # print(x)
    l = ShuffleChannel(p=1.)
    # l.eval()
    out = l(x)
    print(out)
