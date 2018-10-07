import torch
from torch import nn
from torch.nn import functional as F

from dataset import MAX_PARTS, MAX_SENTS, EMBED_DIM

class Task2Vec(nn.Module):
    def __init__(self, embed_dim, conv_config):
        super(Task2Vec, self).__init__()
        
        convs = []
        in_ch = MAX_PARTS
        for cc in conv_config:
            tp = cc.get('type', 'c2')
            if tp == 'c2':
                convs.append(nn.Conv2d(in_channels=in_ch, out_channels=cc['out_ch'], kernel_size=cc['kernel']))
                convs.append(nn.ReLU())
                in_ch=cc['out_ch']
            
        #self.bn = nn.BatchNorm2d(10)
        self.convs = nn.Sequential(*convs)
        self.linear1 = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.linear2 = nn.Linear(in_features=embed_dim, out_features=embed_dim)

    def forward(self, parts):
        data = torch.autograd.Variable(parts)
        #data = self.bn(data)
        conv_out = self.convs.forward(data)
        l1_out = F.relu(self.linear1.forward(conv_out))
        l2_out = self.linear2.forward(l1_out)
        
        return l2_out.view(1,1,-1)
