# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: draw_polygon.py
# @time: 2021/2/20 23:29
# @desc:

from config import *
from PIL import Image, ImageDraw, ImageFont

import os
import json


font = ImageFont.truetype('data/simsun.ttc', 24)

with open(os.path.join(DATAROOT, 'train_labels.json'), 'r', encoding='utf-8') as f:
    label = json.load(f)


img_path = os.path.join(DATAROOT, 'train_images')

for i, key in enumerate(label.keys()):
    if i == 100:
        break
    file = os.path.join(img_path, f'{key}.jpg')

    if i == 0:
        continue

    # import cv2
    # img = cv2.imread(file)
    # h, w = img.shape[:-1]
    # mx = max(h, w)
    # rt = 320. / h if h == mx else 320. / w
    # if mx > 320:
    #     h = int(h * rt)
    #     w = int(w * rt)
    # gmi = cv2.resize(img, (w, h))
    # print(img.shape, gmi.shape)
    # exit()

    if not os.path.exists(file):
        continue
    image = Image.open(file)
    draw = ImageDraw.Draw(image)
    for j in range(len(label[key])):
        points = tuple([tuple(x) for x in label[key][j]['points']])
        text = label[key][j]['transcription']
        draw.polygon(points)
        draw.text((points[0][0], points[0][1]-24), text, font=font)
    image.save(os.path.join(PROJROOT, f'output/{key}.jpg'))



