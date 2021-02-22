# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: loader.py
# @time: 2021/2/21 14:41
# @desc:

from torch.utils.data.dataset import Dataset
from config import *
from utils.pre_process_border import MakeBorderMap
from utils.pre_process_shrink import MakeShrinkMap

import cv2
import copy
import random
import json
import os
import numpy as np
import torch
import torchvision.transforms as transforms


class OCRDataSet(Dataset):
    def __init__(self):
        super(OCRDataSet, self).__init__()
        self.data = []
        self._init_data()
        self._init_preprocessor()
        self._init_transform()

    def _init_data(self):
        with open(os.path.join(DATAROOT, 'train_labels.json'), 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            for key in label_data:
                img_path = os.path.join(DATAROOT, 'train_images', f'{key}.jpg')
                points = [np.array(x['points']) for x in label_data[key]]
                ignores = [x['illegibility'] for x in label_data[key]]
                self.data.append({
                    'img_path': img_path,
                    'text_polys': points,
                    'ignore_tags': ignores,
                })
        print(f'dataset init finished with size: {len(self.data)}')

    def _init_preprocessor(self):
        self._border_maker = MakeBorderMap()
        self._shrink_maker = MakeShrinkMap()

    def _init_transform(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _process(img):
        w, h = img.shape[1], img.shape[0]
        mx = max(h, w)
        rt = float(MAXAXIS) / h if h == mx else float(MAXAXIS) / w
        if mx > MAXAXIS:
            h = int(h * rt)
            w = int(w * rt)
        img = cv2.resize(img, (w, h))
        return img

    def __getitem__(self, index):
        try:
            data = copy.deepcopy(self.data[index])
            img = cv2.imread(data.get('img_path'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self._process(img)
            data['img'] = img
            data['shape'] = [img.shape[0], img.shape[1]]
            data = self._shrink_maker(data)
            data = self._border_maker(data)
            data['img'] = self.transform(data['img'].copy())
            # data['img'] = np.ascontiguousarray(data['img'])
            return data
        except Exception as e:
            LOGGER.exception(e)
            return self.__getitem__(random.randint(0, self.__len__()))

    def __len__(self):
        return len(self.data)

