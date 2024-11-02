# from torchvision import transforms
import cv2
import numpy as np
import types

from PIL import ImageDraw, Image
from numpy import random
# from data.pseudo import PseudoPoints




def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, mask=None):
        for t in self.transforms:
            img, boxes, mask = t(img, boxes, mask)
        return img, boxes, mask


class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, mask=None):
        return self.lambd(img, boxes, mask)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, mask=None):
        return image.astype(np.float32), boxes, mask.astype(np.float32)


class Imgnorm(object):
    def __call__(self, image, boxes=None, mask=None):
        image = image / 255
        return image, boxes, mask




class Resize(object):
    def __init__(self, size=732):
        self.size = size
    def __call__(self, image, boxes=None, mask=None):
        height, width = image.shape
        image = cv2.resize(image, (self.size,self.size))
        mask = cv2.resize(mask, (self.size,self.size))
        boxes[:, 1::2] = boxes[:, 1::2] / height * self.size
        boxes[:, ::2] = boxes[:, ::2] / width * self.size
        return image, boxes, mask


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, image, boxes=None, mask=None):
        alpha = random.uniform(self.lower, self.upper)
        try:
            image *= alpha
        except:
            image = image.astype(np.float32)
            image *= alpha
        return image, boxes, mask


class RandomBrightness(object):
    def __init__(self, delta=50):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    def __call__(self, image, boxes=None, mask=None):
        delta = random.uniform(-self.delta, self.delta)
        try:
            image += delta
        except:
            image = image.astype(np.float32)
            image += delta
        return image, boxes, mask


class BlurAug(object):
    def __call__(self, image, boxes, mask):
        height, width = image.shape
        if random.randint(4) == 0:    # 高斯平滑
            kernel_size = 2 * random.randint(1, 5) + 1
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        if random.randint(2) == 0:  # 高斯噪声
            sigma = random.randint(5, 20)
            gauss = np.random.normal(0, sigma, (height, width))
            image = image + gauss
            image = np.clip(image, a_min=0, a_max=255)
            image = image.astype(np.float32)
        else:  # 椒盐噪声
            s_vs_p = random.uniform(0.1, 0.7)
            amount = random.uniform(0.02, 0.05)
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords[0], coords[1]] = 255
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords[0], coords[1]] = 0
        return image, boxes, mask


class PhotometricDistort(object):
    def __init__(self):
        self.rand_brightness = RandomBrightness()
        self.rand_contrast = RandomContrast()
        self.bluraug = BlurAug()
    def __call__(self, image, boxes, mask):
        x = random.randint(3)
        if x == 0 or x == 1:
            height, width = image.shape
            image, boxes, mask =  self.rand_brightness(image, boxes, mask)
            image, boxes, mask = self.rand_contrast(image, boxes, mask)
            y = random.randint(3)
            if  y== 3 or y == 4:
                image, boxes, mask = self.bluraug(image, boxes, mask)
        return image, boxes, mask


class PseudoBar(object):
    def __init__(self, num_bar=3, w=250, w_h=20, bright_bar=60, thresh = 0.1):
        self.w_h = w_h
        self.w = w
        self.num_bar = num_bar
        self.bright_bar = bright_bar
        self.thresh = thresh

    def __call__(self, image, boxes, mask):
        x = random.randint(5)
        if x == 0 or x == 1 or x == 2:
            height, width = image.shape
            num_bar = np.random.randint(self.num_bar, self.num_bar + 5)
            for i in range(num_bar):
                y = np.random.randint(height)
                x = np.random.randint(width)
                w_h = np.random.randint(self.w_h - 25, self.w_h)
                if random.randint(2):
                    w = np.random.randint(self.w // 2, self.w)
                    h = int(w / w_h)
                else:
                    h = np.random.randint(self.w // 2, self.w)
                    w = int(h / w_h)
                y1 = np.clip(y - h // 2, 0, height)
                y2 = np.clip(y + h // 2, 0, height)
                x1 = np.clip(x - w // 2, 0, width)
                x2 = np.clip(x + w // 2, 0, width)
                rect = np.array([x1, y1, x2, y2])
                overlap = jaccard_numpy(boxes, rect)
                if overlap.max() > self.thresh:
                    i -= 1
                    continue
                if random.randint(2):
                    bright_bar = np.random.randint(250 - self.bright_bar, 250)
                else:
                    bright_bar = np.random.randint(0, self.bright_bar)
                image[y1: y2, x1: x2] = bright_bar
        return image, boxes, mask


class PseudoPoints(object):
    def __init__(self, num_points=15, l=5, h=50, thresh = 0.01):
        self.h = h
        self.l = l
        self.num_points = num_points
        self.thresh = thresh

    def __call__(self, image, boxes, mask):
        height, width = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image.astype(np.uint8))
        draw = ImageDraw.Draw(image)
        x = random.randint(5)
        if x == 0 or x == 1 or x == 2:
            num_points = np.random.randint(self.num_points, self.num_points + 15)
            for i in range(num_points):
                y = np.random.randint(height)
                x = np.random.randint(width)
                r = np.random.randint(self.l, self.h)
                y1 = np.clip(y - r, 0, height)
                y2 = np.clip(y + r, 0, height)
                x1 = np.clip(x - r, 0, width)
                x2 = np.clip(x + r, 0, width)
                rect = np.array([x1, y1, x2, y2])
                overlap = jaccard_numpy(boxes, rect)
                if overlap.max() > self.thresh:
                    i -= 1
                    continue
                bright = np.random.randint(15, 205)
                color = (bright, bright, bright)
                flag = random.randint(2)
                if flag == 0:
                    draw.rectangle((x1, y1, x2, y2), fill=color)
                if flag == 1:
                    draw.ellipse((x1, y1, x2, y2), fill=color)
                # else:
                #     flag1 = random.randint(3)
                #     if flag1 == 1:
                #         draw.polygon(((x1, y1), (x1, y2), (x2, y2)), fill=color)
                #     if flag1 == 2:
                #         draw.polygon(((x1, y1), (x2, y1), (x2, y2)), fill=color)
                #     else:
                #         draw.polygon(((x2, y1), (x1, y2), (x2, y2)), fill=color)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        # image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return image, boxes, mask



class RandomMirror(object):
    def __call__(self, image, boxes, mask):
        height, width = image.shape
        if random.randint(2)==0:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        if random.randint(2) == 0:
            image = image[::-1, :]
            mask = mask[::-1, :]
            boxes[:, 1::2] = height - boxes[:, 3::-2]
        if random.randint(2)==0:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, mask


class ODSegAugment(object):
    def __init__(self, size=512):
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PseudoBar(num_bar=10, w=250, w_h=30, bright_bar=100, thresh=0.02),
            PseudoPoints(),
            PhotometricDistort(),
            RandomMirror(),
            ])

    def __call__(self, img, boxes, mask):
        return self.augment(img, boxes, mask)



class NoAugmentation(object):
    def __init__(self, size=512):
        self.size = size
        self.augment = Resize(self.size)

    def __call__(self, img, boxes, mask):
        return self.augment(img, boxes, mask)




def test():
    from pandas import read_csv
    import torch
    COLOR = [120, 30, 255]
    imgpath = 'E:/YangHR/oganoid/data/gao/img/gao_B7_1_008_9.jpg'
    boxpath = 'E:/YangHR/oganoid/data/gao/box/gao_B7_1_008_9.csv'
    maskpath = 'E:/YangHR/oganoid/data/gao/mask/gao_B7_1_008_9.jpg'
    img = cv2.imdecode(np.fromfile(imgpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(np.fromfile(maskpath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = mask.astype(bool).astype(np.uint8)
    names = ['xmin', 'ymin', 'xmax', 'ymax']
    boxes = read_csv(boxpath, names=names).to_numpy()

    x = ODSegAugment()
    img, box, mask = x(img, boxes, mask)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask * 255
    for pt in boxes:
        cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), COLOR, 2)
        cv2.rectangle(mask, (pt[0], pt[1]), (pt[2], pt[3]), COLOR, 2)

    image = torch.cat((torch.tensor(img), torch.tensor(mask)), dim=1).detach().numpy()
    cv2.imencode('.jpg', image)[1].tofile('C:/Users/Dell/Desktop/aug1.jpg')
    return None


if __name__ == '__main__':
    test()
