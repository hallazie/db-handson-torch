# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: inference.py
# @time: 2021/2/21 21:36
# @desc:

from config import *
from torchvision import transforms
from utils.post_process import SegDetectorRepresenter

import matplotlib.pyplot as plt
import torch
import cv2


transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
postpro = SegDetectorRepresenter()


def process(img):
    w, h = img.shape[1], img.shape[0]
    mx = max(h, w)
    rt = float(MAXAXIS) / h if h == mx else float(MAXAXIS) / w
    if mx > MAXAXIS:
        h = int(h * rt)
        w = int(w * rt)
    img = cv2.resize(img, (w, h))
    return img


def run():
    raw = cv2.imread('data/test-img.jpg')
    model = torch.load('checkpoints/checkpoint-14.pkl').cuda()
    model.eval()
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = process(raw)
    img = torch.unsqueeze(transform(img), 0).cuda()
    pre = model(img).detach().cpu()
    box, sco, seg = postpro({'shape': [[raw.shape[1], raw.shape[0]]]}, pre)

    pre = pre.numpy()
    print(pre.shape)
    plt.subplot(141)
    plt.imshow(pre[0][0])
    plt.title('shrink mask')
    plt.subplot(142)
    plt.imshow(pre[0][1])
    plt.title('border mask')
    plt.subplot(143)
    plt.imshow(raw)
    plt.title('raw image')
    plt.subplot(144)
    plt.imshow(seg[0].numpy())
    plt.title('binary seg')
    plt.show()


def visualize(pre, raw):
    pre = pre.detach().cpu().numpy()
    print(pre.shape)
    plt.subplot(131)
    plt.imshow(pre[0][0])
    plt.title('shrink mask')
    plt.subplot(132)
    plt.imshow(pre[0][1])
    plt.title('border mask')
    plt.subplot(133)
    plt.imshow(raw)
    plt.title('raw image')
    plt.show()


if __name__ == '__main__':
    run()


