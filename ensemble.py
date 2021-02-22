# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: ensemble.py
# @time: 2021/2/21 0:04
# @desc:

from models.head_db import DBHead
from models.neck_fpn import FPN
from models.backbone_resnet import ResNet, BasicBlock

import torchvision.models as models
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2])
        backbone_params = {x[0]: x[1] for x in models.resnet18(pretrained=True).named_parameters()}
        self.backbone.load_state_dict(backbone_params, strict=False)
        self.backbone.cuda()
        self.neck = FPN([64, 128, 256, 512]).cuda()
        self.head = DBHead(256, 256, 2).cuda()

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    img = cv2.imread('output/gt_2.jpg')
    img = cv2.resize(img, (640, 640))
    img = torch.transpose(torch.from_numpy(img), 0, 2).unsqueeze(dim=0).float().cuda()
    print(img.shape)

    detector = Detector()
    res = detector(img)
    print(res.shape)




