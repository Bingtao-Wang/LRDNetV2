# ==============================================================================
# 文件名: utils/data_repro_pipeline.py
# 版本: v2 (Strict Split & Augmentation)
# 改进:
#   1. 严格的数据划分 (按文件名顺序切分，避免时序数据泄露)。
#   2. 训练集加入实时数据增强 (水平翻转)，防止过拟合。
#   3. 修复了虚高评估的数据源问题。
# ==============================================================================

import tensorflow as tf
import os
import glob
import numpy as np
import cv2
from tensorflow.keras.applications.vgg19 import preprocess_input


class AcademicDataGenerator:
    def __init__(self, target='train', batch_size=1, width=1280, height=384,
                 data_dir='data/training'):
        self.target = target
        self.batch_size = batch_size
        self.width = width
        self.height = height

        # 路径配置
        if target == 'test':
            self.image_dir = 'data/testing/image_2'
            self.adi_dir = 'data/testing/ADI'
            self.img_paths = sorted(glob.glob(os.path.join(self.image_dir, '*')))
            self.adi_paths = sorted(glob.glob(os.path.join(self.adi_dir, '*')))
            self.mask_paths = []
        else:
            self.image_dir = os.path.join(data_dir, 'image_2')
            self.adi_dir = os.path.join(data_dir, 'ADI')
            self.mask_dir = os.path.join(data_dir, 'gt_image_2')

            # 1. 获取所有有效的配对文件
            mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '*_road_*.png')))

            all_img_paths = []
            all_adi_paths = []
            all_mask_paths = []

            for m_path in mask_files:
                basename = os.path.basename(m_path)
                parts = basename.split('_')
                # 兼容 KITTI 格式: um_road_000000.png -> um_000000.png
                # 注意：有些文件名可能是 uu_road_000000.png -> uu_000000.png
                img_name = parts[0] + '_' + parts[2]

                img_p = os.path.join(self.image_dir, img_name)
                adi_p = os.path.join(self.adi_dir, img_name)

                if os.path.exists(img_p) and os.path.exists(adi_p):
                    all_img_paths.append(img_p)
                    all_adi_paths.append(adi_p)
                    all_mask_paths.append(m_path)

            # 2. 严格划分 (Strict Split)
            # 不使用随机 Shuffle，而是按顺序截断。
            # KITTI 数据集通常 ID 相邻的图片相似度高。
            # 我们取前 80% 做训练，后 20% 做验证。
            total_samples = len(all_img_paths)
            split_idx = int(total_samples * 0.8)

            if target == 'train':
                self.img_paths = all_img_paths[:split_idx]
                self.adi_paths = all_adi_paths[:split_idx]
                self.mask_paths = all_mask_paths[:split_idx]
                print(f"📘 Training Set: {len(self.img_paths)} samples (First 80%)")
            else:
                self.img_paths = all_img_paths[split_idx:]
                self.adi_paths = all_adi_paths[split_idx:]
                self.mask_paths = all_mask_paths[split_idx:]
                print(f"📙 Validation Set: {len(self.img_paths)} samples (Last 20%)")

    def _sharpen_adi(self, adi_img):
        """复现论文 Eq. 18: ADI 锐化"""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        adi_sharp = cv2.filter2D(adi_img, -1, kernel)
        adi_sharp = cv2.filter2D(adi_sharp, -1, kernel)
        return adi_sharp

    def _read_data(self, img_path, adi_path, mask_path):
        # 1. RGB Image
        img = cv2.imread(img_path.decode('utf-8'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height))
        # 训练时可以在这里加 Augmentation，但为了 CPU 效率，我们移到 TF graph 里做简单的 Flip
        img = preprocess_input(img)

        # 2. ADI Image
        adi = cv2.imread(adi_path.decode('utf-8'))
        adi = cv2.resize(adi, (self.width, self.height))
        adi = self._sharpen_adi(adi)
        adi = adi.astype(np.float32) / 255.0
        if np.any(adi > 0):
            adi = adi - np.mean(adi[adi > 0])

        # 3. Mask
        mask = cv2.imread(mask_path.decode('utf-8'))
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        mask = mask[:, :, 2]
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        return img, adi, mask

    def _tf_map_wrapper(self, img_path, adi_path, mask_path):
        img, adi, mask = tf.numpy_function(
            self._read_data,
            [img_path, adi_path, mask_path],
            [tf.float32, tf.float32, tf.float32]
        )
        img.set_shape([self.height, self.width, 3])
        adi.set_shape([self.height, self.width, 3])
        mask.set_shape([self.height, self.width, 1])

        # ==== 在线数据增强 (Data Augmentation) ====
        # 仅针对训练集 (通过判断 self.target)
        if self.target == 'train':
            # 随机水平翻转 (概率 50%)
            # 注意：Image, ADI, Mask 必须同时翻转！
            if tf.random.uniform(()) > 0.5:
                img = tf.image.flip_left_right(img)
                adi = tf.image.flip_left_right(adi)
                mask = tf.image.flip_left_right(mask)

            # 还可以加一点亮度和对比度增强 (仅对 Image)
            img = tf.image.random_brightness(img, max_delta=0.1)
            # 注意：VGG preprocess 后数值范围变了，brightness 需要小心，这里保守一点先不加颜色变换

        return (img, adi), mask

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths, self.adi_paths, self.mask_paths))

        if self.target == 'train':
            dataset = dataset.shuffle(buffer_size=len(self.img_paths))  # 全量 Shuffle 索引，但不 Shuffle 验证集

        dataset = dataset.map(self._tf_map_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def __len__(self):
        return len(self.img_paths) // self.batch_size
