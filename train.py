import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "-1" 表示隐藏所有显卡，强制使用 CPU
################################### Clean Warning #############
import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import numpy as np

np.seterr(all="ignore")
import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def warn(*args, **kwargs):
    pass


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_AFFINITY"] = "noverbose"

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
################################### Clean Warning #############

from numpy.random import seed
import time
import tensorflow
from tqdm import tqdm
import os
import sys
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from keras.models import load_model
from keras.optimizers import Adam

import keras.backend as K

# Our internal functions and libraries
from models.models import ResearchModels  # For models
from utils.data import DataSet  # For loading datasets
from utils.data_aug import DataSet_aug  # For loading datasets

os.system('cls')


def dice_coef(y_true, y_pred):
    smooth = 1e-5

    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    isct = tf.reduce_sum(y_true * y_pred)

    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def iou_coef(y_true, y_pred):
    smooth = 1e-5
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred):
    return 1.0 - iou_coef(y_true, y_pred)


# Model List: Ours, LUNet, CapsNetR1, CapsNetR3 , CapsNetBasic, Unet
################## Experiment Settings ##################


# legend : ADD= add secondary data, DEPTH for depth, ADI2= for our adi
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
    tensorflow.random.set_seed(seedi)

if 'LRDNet' in model:
    width = 1280
    height = 384
    print('************** Using Size 1280 x 384 **************')

if 'SM' in model:
    width = 256
    height = 256
    print('************** Using Size 256 x 256 **************')

if augmentation:
    train_images = DataSet_aug(model=model, target='train', batch_size=batch_size, width=width, height=height)
    val_images = DataSet_aug(model=model, target='valid', batch_size=batch_size, width=width, height=height)
    aug = '[AUGBIG]'
else:
    train_images = DataSet(model=model, target='train', batch_size=batch_size, width=width, height=height)
    val_images = DataSet(model=model, target='valid', batch_size=batch_size, width=width, height=height)
    aug = ''

steps_per_epoch = train_images.steps_per_epoch
validation_steps = val_images.validation_steps

train_data = train_images.td
val_data = val_images.vd

# ==================== Windows Path Fix Start ====================
# Helper: Save the model (使用 os.path.join 修复 Windows 路径)
checkpoints_dir = os.path.join('results', model)
model_dir = os.path.join(checkpoints_dir, model + '_Weights')
log_parent_dir = os.path.join(checkpoints_dir, model + '_logs')  # 新增：显式定义日志父文件夹

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(log_parent_dir):  # 新增：手动创建日志文件夹，防止 TensorBoard 报错
    os.makedirs(log_parent_dir)

# checkpointer = ModelCheckpoint(filepath=os.path.join(model_dir, model +aug+'.({epoch:03d})-[{loss:.3f},{val_loss:.3f}]-[{val_iou_coef:.4f}].hdf5'),verbose=1,save_best_only=save_best_only)
checkpointer = ModelCheckpoint(
    filepath=os.path.join(model_dir, model + aug + '.({epoch:03d})-[{iou_coef:.4f}]-[{val_iou_coef:.4f}].hdf5'),
    verbose=1, save_best_only=save_best_only)

# Helper: TensorBoard
# 修改：使用已经创建好的 log_parent_dir
tb = TensorBoard(log_dir=os.path.join(log_parent_dir, model))
# ==================== Windows Path Fix End ====================

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.0001, patience=3, verbose=1)

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(os.path.join(checkpoints_dir, model + '_logs', model + str(timestamp) + '.log'))

# helper early stopper
early_stopper = EarlyStopping(monitor='val_iou_coef', patience=patience, verbose=1, baseline=None)
# early_stopper=EarlyStopping(monitor='iou_coef',mode='max', patience=patience, verbose=1)
# Callbacks
if save_models:
    callbacks = [tb, csv_logger, checkpointer]
    # callbacks=[tb, early_stopper, csv_logger,checkpointer]
    # callbacks=[tb, early_stopper, csv_logger,checkpointer,reduce_lr]
else:
    callbacks = [tb, early_stopper, csv_logger, reduce_lr]

# Loading the model
rm = ResearchModels(modelname=model, height=height, width=width)
# Training the Network
history = rm.model.fit_generator(train_data, steps_per_epoch, epochs=epochs, verbose=1,
                                 callbacks=callbacks, validation_data=val_data,
                                 validation_steps=validation_steps)
