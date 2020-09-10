from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable

class ImageFeatureExtraction(nn.Module):
    def __init__(self, use_cuda=True):
        super(ImageFeatureExtraction, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        if use_cuda:
            self.model.cuda()

    def forward(self, image_batch):
        return self.model(image_batch)

class RelationNetwork(nn.Module):
    def __init__(self, n_imgFeat, out_size, qstEmb_size, use_cuda=True):
        super(RelationNetwork, self).__init__()
        self.input_size = (n_imgFeat+2)*2 + qstEmb_size # (256+2)*2 + 11

        self.mlp_g = nn.Sequential(
            nn.Linear(self.input_size, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 2000),
            nn.ReLU(inplace=True)
        )

        self.mlp_f = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_size),
            nn.LogSoftmax(dim=1)
        )

        if use_cuda:
            self.mlp_g.cuda()
            self.mlp_f.cuda()

    def forward(self, imgFeat, qstEmb):
        b, n_obj, d_obj = imgFeat.size() # Image feature size [batch x 5*5 x (256+2)]
        d_qst = qstEmb.size(1) # Question embedding size [batch x 11]

        # Make all possible pairs (o_i, o_j, q)
        o_i = imgFeat.unsqueeze(1).expand([b, n_obj, n_obj, d_obj]) # [batch x 5*5 x 5*5 x (256+2)]
        o_j = imgFeat.unsqueeze(2).expand([b, n_obj, n_obj, d_obj]) # [batch x 5*5 x 5*5 x (256+2)]
        q = qstEmb.unsqueeze(1).unsqueeze(2).expand([b, n_obj, n_obj, d_qst])  # [batch x 5*5 x 5*5 x 11]

        in_g = torch.cat([o_i,o_j,q], 3).view(b*n_obj*n_obj, (d_obj*2)+d_qst) # [batch*5*5*5*5 x (256+2)*2+11]

        # Feed forward to 'g'
        relation = self.mlp_g(in_g) # [batch*5*5*5*5 x 2000]
        relation = relation.view(b, n_obj*n_obj, 2000) # [batch x 5*5*5*5 x 2000]
        relation = relation.sum(1).squeeze(1) # [batch x 2000]

        # Feed forward to 'f'
        out = self.mlp_f(relation)
        return out

class net(nn.Module):
    def __init__(self, question_len=11, n_feature=256+2, n_classes=10, use_cuda=True):
        super(net, self).__init__()
        self.use_cuda = use_cuda

        self.ImageFeatrueExtraction = ImageFeatureExtraction(use_cuda=use_cuda)
        self.RelationalNetwork = RelationNetwork(n_imgFeat=n_feature, out_size=n_classes,
                                                 qstEmb_size=question_len, use_cuda=use_cuda)
        self.build_coord_tensor(64, 5)

    def forward(self, image_batch, question_batch):
        imgFeat = self.ImageFeatrueExtraction(image_batch)  # [batch x 256 x 5 x 5]

        b,c,d,_ = imgFeat.size()
        imgFeat = imgFeat.view(b, c, d*d) # [batch x 256 x 5*5]

        # Tag coordinates to the image features
        if self.coord_tensor.size()[0] != b:  # In case batch-size changes
            self.build_coord_tensor(b, d)
        self.coord_tensor = self.coord_tensor.view(b, 2, d * d)  # [batch x 2 x 5*5]

        imgFeat = torch.cat([imgFeat, self.coord_tensor], 1)  # [batch x (256+2) x 5*5]
        imgFeat = imgFeat.permute(0, 2, 1)  # [batch x 5*5 x (256+2)]

        out = self.RelationalNetwork(imgFeat, question_batch)
        return out

    def build_coord_tensor(self, b, d):
        """
        This part gently borrowed from
        https://github.com/mesnico/RelationNetworks-CLEVR/blob/master/model.py
        """
        coords = torch.linspace(-d / 2., d / 2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x, y))
        ct = ct.unsqueeze(0).repeat(b, 1, 1, 1)
        self.coord_tensor = Variable(ct, requires_grad=False)

        if self.use_cuda:
            self.coord_tensor = self.coord_tensor.cuda()
