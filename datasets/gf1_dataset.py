import os
import torch
from torch.utils.data import Dataset
import numpy as np

class GF1_cls_FULL(Dataset):

    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')

        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        block_dir = os.path.join(voc_root, 'block_label/bl_npy/')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".npy") for x in file_names]
        self.blocks = [os.path.join(block_dir, x + ".npy") for x in file_names]
        assert (len(self.images) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):

        name = self.images[index]

        # get the hyperspectral data and lables
        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)
        img = torch.tensor(rsData[:,:320,:320])

        block_label = np.load(self.blocks[index])
        block_label = np.expand_dims(block_label, axis=0)
        block_label = torch.tensor(block_label)

        mask = np.load(self.masks[index])
        mask = np.expand_dims(mask, axis=0)
        target = torch.tensor(mask[:, :320,:320])

        return (img, block_label, target, name)

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)



class GF1_cls_WEAK(Dataset):

    def __init__(self, root, image_set='train'):

        self.root = os.path.expanduser(root)
        self.image_set = image_set

        voc_root = self.root

        image_dir = os.path.join(voc_root, 'JPEGImages')
        TOA_dir = os.path.join(voc_root, 'JPEGImages_TOA')
        block_dir = os.path.join(voc_root, 'block_label/bl_npy/')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".npy") for x in file_names]
        self.toas = [os.path.join(TOA_dir, x + ".npy") for x in file_names]
        self.blocks = [os.path.join(block_dir, x + ".npy") for x in file_names]
        assert (len(self.images) == len(self.blocks))

    # just return the img and target in P[key]
    def __getitem__(self, index):

        name = self.images[index]

        # get the hyperspectral data and lables
        rsData = np.load(self.images[index])
        rsData = np.asarray(rsData, dtype=np.float32)
        rsData = rsData.transpose(2, 0, 1)
        img = torch.tensor(rsData[:,:320,:320])


        toaData = np.load(self.toas[index])
        HOT = toaData[:, :, 0] - 0.5 * toaData[:, :, 2]
        HOT = np.expand_dims(HOT, axis=0)
        HOT = torch.tensor(HOT[:, :320,:320])

        block_label = np.load(self.blocks[index])
        block_label = np.expand_dims(block_label, axis=0)
        block_label = torch.tensor(block_label)

        return (img, block_label, name, HOT)

    # return the amount of images（train set）
    def __len__(self):
        return len(self.images)


