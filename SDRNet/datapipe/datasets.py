import random
import numpy as np
from pathlib import Path
from scipy.io import loadmat

import cv2
import torch
from functools import partial
import torchvision as thv
from torch.utils.data import Dataset

from SDRNet.utils import util_sisr
from SDRNet.utils import util_image
from SDRNet.utils import util_common


class LamaDistortionTransform:
    def __init__(self, kwargs):
        import albumentations as A
        from .aug import IAAAffine2, IAAPerspective2
        out_size = kwargs.get('pch_size', 256)
        self.transform = A.Compose([
            A.SmallestMaxSize(max_size=out_size),
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.Normalize(mean=kwargs.mean, std=kwargs.std, max_pixel_value=kwargs.max_value),
        ])

    def __call__(self, im):
        '''
        im: numpy array, h x w x c, [0,1]

        '''
        return self.transform(image=im)['image']

def get_transforms(transform_type, kwargs):
    '''
    Accepted optins in kwargs.
        mean: scaler or sequence, for nornmalization
        std: scaler or sequence, for nornmalization
        crop_size: int or sequence, random or center cropping
        scale, out_shape: for Bicubic
        min_max: tuple or list with length 2, for cliping
    '''
    if transform_type == 'default':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None), out_shape=kwargs.get('out_shape', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_back_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None)),
            util_sisr.Bicubic(scale=1/kwargs.get('scale', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'resize_ccrop_norm':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            # max edge resize if crop_size is int
            thv.transforms.Resize(size=kwargs.get('size', None)),
            thv.transforms.CenterCrop(size=kwargs.get('size', None)),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'rcrop_aug_norm':
        transform = thv.transforms.Compose([
            util_image.RandomCrop(pch_size=kwargs.get('pch_size', 256)),
            util_image.SpatialAug(
                only_hflip=kwargs.get('only_hflip', False),
                only_vflip=kwargs.get('only_vflip', False),
                only_hvflip=kwargs.get('only_hvflip', False),
                ),
            util_image.ToTensor(max_value=kwargs.get('max_value')),  # (ndarray, hwc) --> (Tensor, chw)
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'aug_norm':
        transform = thv.transforms.Compose([
            util_image.SpatialAug(
                only_hflip=kwargs.get('only_hflip', False),
                only_vflip=kwargs.get('only_vflip', False),
                only_hvflip=kwargs.get('only_hvflip', False),
                ),
            util_image.ToTensor(),   # hwc --> chw
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'lama_distortions':
        transform = thv.transforms.Compose([
                LamaDistortionTransform(kwargs),
                util_image.ToTensor(max_value=1.0),   # hwc --> chw
            ])
    elif transform_type == 'rgb2gray':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),   # c x h x w, [0,1]
            thv.transforms.Grayscale(num_output_channels=kwargs.get('num_output_channels', 3)),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

def create_dataset(dataset_config):
    if dataset_config['type'] == 'base':
        dataset = BaseData(**dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

class BaseData(Dataset):
    def __init__(
            self,
            dir_path,
            txt_path=None,
            transform_type='default',
            transform_kwargs={'mean':0.0, 'std':1.0},
            extra_dir_path=None,
            extra_transform_type=None,
            extra_transform_kwargs=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = []
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))
        if txt_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_path))

        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)

        self.extra_dir_path = extra_dir_path
        if extra_dir_path is not None:
            assert extra_transform_type is not None
            self.extra_transform = get_transforms(extra_transform_type, extra_transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path_base = self.file_paths[index]
        im_base = util_image.imread(im_path_base, chn='rgb', dtype='float32')

        im_target = self.transform(im_base)
        out = {'image':im_target, 'lq':im_target}

        if self.extra_dir_path is not None:
            im_path_extra = Path(self.extra_dir_path) / Path(im_path_base).name
            im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='float32')
            im_extra = self.extra_transform(im_extra)
            out['gt'] = im_extra

        if self.need_path:
            out['path'] = im_path_base

        return out

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

