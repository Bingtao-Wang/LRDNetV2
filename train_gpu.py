import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # 降低 TF 日志
os.environ["KMP_AFFINITY"] = "noverbose"

# ------------------- Clean warnings -------------------
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
np.seterr(all="ignore")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(3)

# GPU 显存按需增长（40 系推荐）
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
# ------------------------------------------------------

from numpy.random import seed
import time
from tqdm import tqdm
import sys

# ==== 统一改用 tf.keras ====
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Our internal functions and libraries
from models.models import ResearchModels         # For models
from utils.data import DataSet                   # For loading datasets
from utils.data_aug import DataSet_aug           # For loading datasets

# ----------------- metrics & losses -------------------
def dice_coef(y_true, y_pred):
    smooth = 1e-5
    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))
    isct = tf.reduce_sum(y_true * y_pred)
    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def iou_coef(y_true, y_pred):
    smooth = 1e-5
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, [1,2,3]) + K.sum(y_pred, [1,2,3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def iou_loss(y_true, y_pred):
    return 1.0 - iou_coef(y_true, y_pred)
# ------------------------------------------------------

# ================= Experiment Settings =================
model = 'LRDNet_Test_SM'
augmentation = False
save_best_only = True
seeding = False
batch_size = 1
patience = 15
epochs = 1500
save_models = True

if seeding:
    seedi = 100
    seed(seedi)
    tf.random.set_seed(seedi)

# 根据模型名选尺寸
if 'LRDNet' in model:
    width, height = 1280, 384
    print('************** Using Size 1280 x 384 **************')

if 'SM' in model:
    width, height = 256, 256
    print('************** Using Size 256 x 256 **************')

# 数据管道（保持与原仓库一致）
if augmentation:
    train_images = DataSet_aug(model=model, target='train', batch_size=batch_size, width=width, height=height)
    val_images   = DataSet_aug(model=model, target='valid', batch_size=batch_size, width=width, height=height)
    aug = '[AUGBIG]'
else:
    train_images = DataSet(model=model, target='train', batch_size=batch_size, width=width, height=height)
    val_images   = DataSet(model=model, target='valid', batch_size=batch_size, width=width, height=height)
    aug = ''

steps_per_epoch   = train_images.steps_per_epoch
validation_steps  = val_images.validation_steps
train_data        = train_images.td   # 期望是 tf.keras.utils.Sequence 或 Python generator
val_data          = val_images.vd

# 目录
checkpoints_dir = os.path.join('results', model)
model_dir = os.path.join(checkpoints_dir, model + '_Weights')
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 回调
ckpt_name = os.path.join(model_dir, model + aug + '.({epoch:03d})-[{iou_coef:.4f}]-[{val_iou_coef:.4f}].hdf5')
checkpointer = ModelCheckpoint(
    filepath=ckpt_name,
    verbose=1,
    save_best_only=save_best_only,
    monitor='val_iou_coef',
    mode='max'
)

tb = TensorBoard(log_dir=os.path.join(checkpoints_dir, model + '_logs', model))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.0001, patience=3, verbose=1)
timestamp = time.time()
csv_logger = CSVLogger(os.path.join(checkpoints_dir, model + '_logs', f'{model}{timestamp}.log'))

# 也可以启用早停（原脚本默认注释）
early_stopper = EarlyStopping(monitor='val_iou_coef', mode='max', patience=patience, verbose=1)

if save_models:
    callbacks = [tb, csv_logger, checkpointer, reduce_lr]
else:
    callbacks = [tb, csv_logger, early_stopper, reduce_lr]

# 模型
rm = ResearchModels(modelname=model, height=height, width=width)

# ⚠️ 确保 ResearchModels 在 compile 时注册我们的自定义 metrics（否则监控不到）
# 例如：
# rm.model.compile(optimizer=Adam(...), loss=iou_loss, metrics=[iou_coef, dice_coef])

# 训练（TF2 中推荐统一用 .fit）
history = rm.model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_data,
    validation_steps=validation_steps
)
