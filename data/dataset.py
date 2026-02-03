from PIL import Image
from torch.utils.data import Dataset
import data.util as Util
from model.ColorChannelCompensation import three_c as t_c
import numpy as np
import cv2


def MutiScaleLuminanceEstimation(img):
    sigma_list = [15, 60, 90]
    w, h, c = img.shape
    img = cv2.resize(img, dsize=None, fx=0.3, fy=0.3)
    Luminance = np.ones_like(img).astype(float)
    for sigma in sigma_list:
        Luminance1 = np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
        Luminance1 = np.clip(Luminance1, 0, 255)
        Luminance += Luminance1
    Luminance = Luminance / 3
    L = (Luminance - np.min(Luminance)) / (np.max(Luminance) - np.min(Luminance) + 0.0001)
    L = np.uint8(L * 255)
    L = cv2.resize(L, dsize=(h, w))
    return L


class UIEDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.data_len = data_len
        self.split = split

        self.input_path = Util.get_paths_from_images('{}/input_{}'.format(dataroot, resolution))
        self.target_path = Util.get_paths_from_images('{}/target_{}'.format(dataroot, resolution))
        self.depth_path_root = f'{dataroot}/target_depth_b/'

        self.dataset_len = len(self.target_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        # print(f'{split}:{self.data_len}')

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        name = self.target_path[index].split('/')[-1]
        depth_name = name.split('.')[0] + '_depth.png'
        target = Image.open(self.target_path[index]).convert("RGB")
        input = Image.open(self.input_path[index]).convert("RGB")
        depth = cv2.imread(self.depth_path_root + depth_name, cv2.IMREAD_GRAYSCALE)
        # input = t_c(input)
        [input, target] = Util.transform_augment([input, target], split=self.split, min_max=(-1, 1))
        depth_normalized = cv2.convertScaleAbs(depth, alpha=255.0 / depth.max())
        [depth_normalized] = Util.transform_augment([depth_normalized], split=self.split, min_max=(-1, 1))
        # print(depth_normalized)

        input = np.array(input).astype(np.float32)
        target = np.array(target).astype(np.float32)
        depth_normalized = np.array(depth_normalized).astype(np.float32)

        return {'target': target, 'input': input, 'Index': index, 'name': name, 'depth': - depth_normalized}


class UIEDatasetDDIM(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.data_len = data_len
        self.split = split

        self.data_path = Util.get_paths_from_images('{}/input'.format(dataroot))

        self.dataset_len = len(self.data_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        # print(f'{split}:{self.data_len}')

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img = Image.open(self.data_path[index]).convert("RGB")
        # img = t_c(img)
        [img] = Util.transform_augment([img], split=self.split, min_max=(-1, 1))

        img = np.array(img).astype(np.float32)

        name = self.data_path[index].split('/')[-1]
        return {'input': img, 'Index': index, 'name': name}



