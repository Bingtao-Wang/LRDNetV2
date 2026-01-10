import random
from PIL import Image
import numpy as np
import pandas as pd
import imageio
import os.path
import re
# 【已修改】使用 tensorflow.keras
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import albumentations as A
import cv2 as cv
import glob


class DataSet():
    def __init__(self, model=None, target='train', batch_size=2, width=None, height=None, dim=3, filled=False):

        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.dim = dim
        self.target = target
        self.model = model

        validation = 30  # validation split percentage
        validation = validation / 100

        # Path setting for dataset
        self.image_data_folder = 'data/training/image_2/'
        self.ADI_folder = 'data/training/ADI/'
        self.mask_data_folder = 'data/training/gt_image_2/'
        self.test_dir = "data/testing/image_2/"
        self.test_dir_ADI = "data/testing/ADI/"

        # List of testimages
        if os.path.exists(self.test_dir):
            test_images = os.listdir(self.test_dir)
            test_images.sort()
            test_images = np.array(test_images)
        else:
            test_images = []

        if os.path.exists(self.test_dir_ADI):
            test_images_ADI = os.listdir(self.test_dir_ADI)
            test_images_ADI.sort()
            test_images_ADI = np.array(test_images_ADI)
        else:
            test_images_ADI = []

        # List of images ad their masks for training
        uu_all_images = glob.glob(self.image_data_folder + 'uu_*.*')
        uu_all_images.sort()
        uu_all_images = np.array(uu_all_images)

        um_all_images = glob.glob(self.image_data_folder + 'um_*.*')
        um_all_images.sort()
        um_all_images = np.array(um_all_images)

        umm_all_images = glob.glob(self.image_data_folder + 'umm_*.*')
        umm_all_images.sort()
        umm_all_images = np.array(umm_all_images)

        um = []
        uu = []
        umm = []

        # ================= Fix Windows Path Issue =================
        for x in uu_all_images:
            uu.append(os.path.basename(x))

        for x in um_all_images:
            um.append(os.path.basename(x))

        for x in umm_all_images:
            umm.append(os.path.basename(x))
        # ==========================================================

        random.shuffle(um)
        random.shuffle(umm)
        random.shuffle(uu)

        n_um = round(uu_all_images.size * validation)
        n_umm = round(um_all_images.size * validation)
        n_uu = round(umm_all_images.size * validation)

        self.train_images = um[:um_all_images.size - n_um]
        self.valid_images = um[um_all_images.size - n_um:]

        self.train_images = np.append(self.train_images, umm[:umm_all_images.size - n_umm])
        self.train_images = np.append(self.train_images, uu[:uu_all_images.size - n_uu])

        self.valid_images = np.append(self.valid_images, uu[1 + umm_all_images.size - n_umm:])
        self.valid_images = np.append(self.valid_images, uu[1 + uu_all_images.size - n_uu:])

        self.all_mask_images = {}

        if self.target == 'train':
            self.td = self.batch_generator(self.train_images, self.batch_size, target=self.target)
            self.steps_per_epoch = self.train_images.shape[0] / self.batch_size
            return None
        if self.target == 'valid':
            self.vd = self.batch_generator(self.valid_images, self.batch_size, target=self.target)
            self.validation_steps = self.valid_images.shape[0] / self.batch_size
            return None
        if self.target == 'test':
            self.test_images = test_images
            self.test_images_ADI = test_images_ADI
            return None

    def batch_generator(self, images, batch_size, target='train'):
        dims = [self.width, self.height]

        # 定义 transform (保持原样)
        transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.75, rotate_limit=45, p=0.5),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(),
            ], p=0.2),
            A.OneOf([
                A.GridDistortion(p=1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
            ], p=0.2),
            A.OneOf([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.VerticalFlip(p=0.5)
            ], p=0.2),
            A.OneOf([
                A.RandomResizedCrop(self.height, self.width, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333),
                                    interpolation=1, always_apply=False, p=1.0),
                A.PadIfNeeded(min_height=self.height, min_width=self.width, p=1)
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(p=0.2),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.3)
            ], p=0.2),
        ], additional_targets={'mask0': 'mask'})

        transform2 = A.Compose([], additional_targets={'mask0': 'mask'})

        while True:
            if images.shape[0] == 0:
                print("Warning: No images found for generator.")
                break

            ix = np.random.choice(np.arange(images.shape[0]), batch_size)
            batch_features = []
            batch_features_2 = []
            batch_labels = []

            index = 0
            for i in ix:
                if (np.random.uniform(0, 1) > 0.5):
                    flip = 1
                else:
                    flip = 0

                # --- 加载 image ---
                original_img = Image.open(self.image_data_folder + images[i])
                resized_img = original_img.resize((self.width, self.height))
                image0 = img_to_array(resized_img) / 255.

                # --- 加载 ADI ---
                original_img = Image.open(self.ADI_folder + images[i]).convert('RGB')
                resized_img = original_img.resize((self.width, self.height))
                image1 = img_to_array(resized_img) / 255.

                image1 = np.array(image1)
                # 修复: 避免全0矩阵计算 mean 报错
                if np.any(image1 > 0):
                    image1 = image1 - np.mean(image1[image1 > 0])
                else:
                    image1 = image1  # 全0不需要减均值

                # --- 加载 mask ---
                fname_parts = images[i].split('_')
                mask_name = fname_parts[0] + '_road_' + fname_parts[1]
                original_mask = load_img(self.mask_data_folder + mask_name)
                resized_mask = original_mask.resize((self.width, self.height))
                label = img_to_array(resized_mask) / 255.
                mask = label[:, :, 2]

                transformed = transform(image=image0, mask=image1, mask0=mask)

                if target == 'train':
                    if flip == 1:
                        batch_features.append(transformed['image'])
                        batch_features_2.append(transformed['mask'])
                        batch_labels.append(transformed['mask0'])
                    else:
                        batch_features.append(image0)
                        batch_features_2.append(image1)
                        batch_labels.append(label[:, :, 2])
                else:
                    batch_features.append(image0)
                    batch_features_2.append(image1)
                    batch_labels.append(label[:, :, 2])

                index = index + 1

            # 【修复点】将 numpy 转换逻辑移出 for 循环
            batch_features = np.array(batch_features, dtype='float32')
            batch_features_2 = np.array(batch_features_2, dtype='float32')
            batch_labels = np.array(batch_labels, dtype='float32')
            batch_labels = batch_labels[:, :, :, np.newaxis]

            yield [batch_features, batch_features_2], batch_labels
