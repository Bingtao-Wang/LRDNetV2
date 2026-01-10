import os
import sys
import time
import warnings
import logging
import numpy as np
from tqdm import tqdm

# ------------------- 1. 核心 GPU 与框架设置 (必须放在最前面) -------------------
# 设置 segmentation_models 使用 tf.keras (关键！解决 keras 和 tf.keras 冲突)
os.environ["SM_FRAMEWORK"] = "tf.keras"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确保这里是你的显卡ID
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

# ---- GPU 显存按需增长 & 混合精度设置 ----
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"✅ 发现 GPU: {gpus}")
        print("✅ 已开启显存按需分配")
    except RuntimeError as e:
        print(e)

# 开启混合精度 (Mixed Precision) - 极大提升 30/40 系显卡速度
# try:
#     policy = mixed_precision.Policy('mixed_float16')
#     mixed_precision.set_global_policy(policy)
#     print("✅ 已开启混合精度训练 (mixed_float16)")
# except Exception as e:
#     print("⚠️ 混合精度开启失败，将使用默认精度:", e)
# --------------------------------------------------------------------------

# ------------------- Clean warnings -------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(all="ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
# ------------------------------------------------------

from numpy.random import seed

# ==== 统一改用 tf.keras ====
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam

# Our internal functions and libraries
from models.models import ResearchModels
from utils.data import DataSet
from utils.data_aug import DataSet_aug


# ----------------- metrics & losses -------------------
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # 混合精度下，确保计算使用 float32 以避免数值溢出
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iou_coef(y_true, y_pred):
    smooth = 1e-5
    # 强制转换为 float32 进行计算
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred):
    return 1.0 - iou_coef(y_true, y_pred)


# ------------------------------------------------------

# ================= Experiment Settings =================
model_name = 'LRDNet_Test_SM_TF2'  # 避免变量名覆盖 model 对象
augmentation = False
save_best_only = True
seeding = False
batch_size = 8  # GPU 上通常可以比 CPU 开大一点，如果是 4090 可以尝试 8 或 16
patience = 15
epochs = 1500
save_models = True

if seeding:
    seedi = 100
    seed(seedi)
    tf.random.set_seed(seedi)

# 根据模型名选尺寸
if 'LRDNet' in model_name:
    width, height = 1280, 384
    print('************** Using Size 1280 x 384 **************')
if 'SM' in model_name:
    width, height = 256, 256
    print('************** Using Size 256 x 256 **************')

# 数据管道
if augmentation:
    train_images = DataSet_aug(model=model_name, target='train', batch_size=batch_size, width=width, height=height)
    val_images = DataSet_aug(model=model_name, target='valid', batch_size=batch_size, width=width, height=height)
    aug = '[AUGBIG]'
else:
    train_images = DataSet(model=model_name, target='train', batch_size=batch_size, width=width, height=height)
    val_images = DataSet(model=model_name, target='valid', batch_size=batch_size, width=width, height=height)
    aug = ''

steps_per_epoch = train_images.steps_per_epoch
validation_steps = val_images.validation_steps
train_data = train_images.td
val_data = val_images.vd

# 目录
checkpoints_dir = os.path.join('results', model_name)
model_dir = os.path.join(checkpoints_dir, model_name + '_Weights')
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 回调
ckpt_name = os.path.join(model_dir, model_name + aug + '.({epoch:03d})-[{iou_coef:.4f}]-[{val_iou_coef:.4f}].hdf5')
checkpointer = ModelCheckpoint(
    filepath=ckpt_name,
    verbose=1,
    save_best_only=save_best_only,
    monitor='val_iou_coef',
    mode='max'
)

tb = TensorBoard(log_dir=os.path.join(checkpoints_dir, model_name + '_logs', model_name))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7)
timestamp = time.time()
csv_logger = CSVLogger(os.path.join(checkpoints_dir, model_name + '_logs', f'{model_name}{timestamp}.log'))
early_stopper = EarlyStopping(monitor='val_iou_coef', mode='max', patience=patience, verbose=1)

callbacks = [tb, csv_logger, checkpointer, reduce_lr]
if not save_models:
    callbacks.append(early_stopper)

# 模型构建
# 注意：ResearchModels 内部会因为 SM_FRAMEWORK 的设置而正确调用 tf.keras
rm = ResearchModels(modelname=model_name, height=height, width=width)

# 编译模型
# 调整：混合精度下，epsilon 需要调整以保持稳定性，但在 Adam 默认参数通常没问题
rm.model.compile(optimizer=Adam(learning_rate=5e-5), loss=iou_loss, metrics=[iou_coef])

print(f"🚀 开始训练: Batch Size = {batch_size}, Epochs = {epochs}")

history = rm.model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_data,
    validation_steps=validation_steps
)
