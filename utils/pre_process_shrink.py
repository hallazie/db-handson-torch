# --*-- coding:utf-8 --*--

import numpy as np
import cv2


def shrink_polygon_py(polygon, shrink_ratio):
    """
    对框进行缩放，返回去的比例为1/shrink_ratio 即可
    """
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()
    polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
    polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
    return polygon


def shrink_polygon_pyclipper(polygon, shrink_ratio):
    from shapely.geometry import Polygon
    import pyclipper
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if not shrinked:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked


class MakeShrinkMap:
    """
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    """

    def __init__(self, min_text_size=8, shrink_ratio=0.4, shrink_type='pyclipper'):
        shrink_func_dict = {'py': shrink_polygon_py, 'pyclipper': shrink_polygon_pyclipper}
        self.shrink_func = shrink_func_dict[shrink_type]
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                shrinked = self.shrink_func(polygon, self.shrink_ratio)
                if shrinked.size == 0:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        """
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        """
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    @staticmethod
    def polygon_area(pg):
        return cv2.contourArea(pg)


if __name__ == '__main__':
    from shapely.geometry import Polygon
    import pyclipper

    polygon = np.array([[0, 0], [100, 10], [100, 100], [10, 90]])
    a = shrink_polygon_py(polygon, 0.4)
    print('a', a)
    print('-----------')
    print('shrink', shrink_polygon_py(a, 1 / 0.4))
    print('-----------')
    b = shrink_polygon_pyclipper(polygon, 0.4)
    print('b', b)
    print('-----------')
    poly = Polygon(b)
    distance = poly.area * 1.5 / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(b, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    bounding_box = cv2.minAreaRect(expanded)
    points = cv2.boxPoints(bounding_box)
    print('points', points)
    print(a.shape, b.shape, points.shape)

    import os
    import json
    import matplotlib.pyplot as plt
    import logging
    from config import *
    from PIL import Image
    from utils.pre_process_border import MakeBorderMap
    from matplotlib.pyplot import figure

    logger = logging.getLogger('none')

    figure(num=None, figsize=(16, 12), dpi=312, facecolor='w', edgecolor='k')

    maker_s = MakeShrinkMap()
    maker_b = MakeBorderMap()

    with open(os.path.join(DATAROOT, 'train_labels.json'), 'r', encoding='utf-8') as f:
        label = json.load(f)

    img_path = os.path.join(DATAROOT, 'train_images')

    for i, key in enumerate(label.keys()):
        if i == 100:
            break
        try:
            file = os.path.join(img_path, f'{key}.jpg')
            points = [np.array(x['points']) for x in label[key]]
            ignores = [x['illegibility'] for x in label[key]]
            if not os.path.exists(file):
                continue
            image = np.array(Image.open(file))
            data = {
                'img': image,
                'text_polys': points,
                'ignore_tags': ignores,
            }
            data = maker_b(data)
            data = maker_s(data)
            plt.subplot(231)
            plt.imshow(data['shrink_map'])
            plt.title('shrink_map')
            plt.subplot(232)
            plt.imshow(data['shrink_mask'])
            plt.title('shrink_mask')
            plt.subplot(234)
            plt.imshow(data['threshold_map'])
            plt.title('threshold_map')
            plt.subplot(235)
            plt.imshow(data['threshold_mask'])
            plt.title('threshold_mask')
            plt.subplot(233)
            plt.imshow(image)
            plt.title('raw')
            # plt.show()
            plt.savefig(os.path.join(PROJROOT, 'output', f'{key}-plot.png'))
            plt.cla()
        except Exception as e:
            logger.exception(e)




