import torch
import random
import imageio
import numpy as np

from torchvision import transforms


def top_left_crop(vid, output_size):
    th, tw = output_size
    return crop(vid, 0, 0, th, tw)


def top_right_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = 0
    j = int(round((w - tw)))
    return crop(vid, i, j, th, tw)


def bottom_right_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th)))
    j = int(round((w - tw)))
    return crop(vid, i, j, th, tw)


def bottom_left_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th)))
    j = 0
    return crop(vid, i, j, th, tw)


def get_transform(is_validation, crop_size=112):
    size = 128 if crop_size == 112 else 256
    if is_validation:
        crop = CenterCrop((crop_size, crop_size))
    else:
        crop = RandomCrop((crop_size, crop_size))
    normalize = Normalize(mean=[0.43216, 0.394666, 0.37645],
                          std=[0.22803, 0.22145, 0.216989])
    transform = [
        ToFloatTensorInZeroOne(),
        Resize(size),
        normalize,
        crop]
    if not is_validation:
        transform += [RandomHorizontalFlip()]
    return transforms.Compose(transform)


def inverse_transform(buffer):
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]
    images = buffer.cpu().numpy()
    images = np.stack([im * s for im, s in zip(images, std)])
    images = np.stack([im + m for im, m in zip(images, mean)])
    images = (images * 255).astype('uint8')
    images = images.transpose([1, 2, 3, 0])
    return images


def batch2gif(buffer, label, savepath, classes=None):
    images = inverse_transform(buffer)
    if classes is not None:
        name = classes[int(label)]
    else:
        name = ''
    imageio.mimsave(savepath+name+'.gif', images)


def crop(vid, i, j, h, w):
    return vid[..., i:(i + h), j:(j + w)]


def center_crop(vid, output_size):
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(vid, i, j, th, tw)


def hflip(vid):
    return vid.flip(dims=(-1,))


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def resize(vid, size, interpolation='bilinear'):
    # NOTE: using bilinear interpolation because we don't work on minibatches
    # at this level
    scale = None
    if isinstance(size, int):
        scale = float(size) / min(vid.shape[-2:])
        size = None
    return torch.nn.functional.interpolate(
        vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)


def pad(vid, padding, fill=0, padding_mode="constant"):
    # NOTE: don't want to pad on temporal dimension, so let as non-batch
    # (4d) before padding. This works as expected
    return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)


def to_normalized_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255

def to_float_tensor(vid):
    return vid.permute(3, 0, 1, 2).to(torch.float32)


def normalize(vid, mean, std):
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean = torch.as_tensor(mean).reshape(shape)
    std = torch.as_tensor(std).reshape(shape)
    return (vid - mean) / std


# Class interface

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(vid, output_size):
        """Get parameters for ``crop`` for a random crop.
        """
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid):
        i, j, h, w = self.get_params(vid, self.size)
        return crop(vid, i, j, h, w)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return center_crop(vid, self.size)


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        return resize(vid, self.size)


class ToFloatTensorInZeroOne(object):
    def __call__(self, vid):
        return to_normalized_float_tensor(vid)


class ToFloatTensor(object):
    def __call__(self, vid):
        return to_float_tensor(vid)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, vid):
        return normalize(vid, self.mean, self.std)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, vid):
        if random.random() < self.p:
            return vid.flip(dims=(-1,))#hflip(vid)
        return vid


class Pad(object):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, vid):
        return pad(vid, self.padding, self.fill)