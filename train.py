# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: train.py
# @time: 2021/2/21 17:57
# @desc:

from loader.loader import OCRDataSet
from loss.loss_db import DBLoss

import torch
import torch.optim as optim
import warnings


warnings.filterwarnings("ignore")


def train():
    epochs = 100
    lr = 1e-5

    # detector = Detector()
    # detector.train()

    detector = torch.load('checkpoints/checkpoint-base.pkl').cuda()
    detector.train()

    dataset = OCRDataSet()
    criterion = DBLoss()
    optimizer = optim.Adam(detector.parameters(), lr=lr)

    for ep in range(epochs):
        for bidx in range(dataset.__len__()):
            try:
                batch = dataset.__getitem__(bidx)
                optimizer.zero_grad()
                pred = detector(torch.unsqueeze(batch['img'], 0).cuda())
                loss = criterion(pred, batch)
                loss['loss'].backward()
                optimizer.step()
                if bidx % 50 == 0:
                    print(f'epoch={ep}, batch={bidx}, loss: {loss}')
            except Exception as e:
                print(e)
                break
        torch.save(detector, f'checkpoints/checkpoint-{ep}.pkl')


if __name__ == '__main__':
    train()
