# ==============================================================================
# 文件名: train_repro_academic.py
# 创建目的:
#   1. LRDNet 论文 Baseline 复现主程序。
#   2. 包含学术级评估指标 (Precision, Recall, F1-Score)。
#   3. 修复了 OOM 问题 (调整 Batch Size)。
#   4. 使用 mixed_float16 进行加速。
# ==============================================================================

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
from tensorflow.keras.optimizers import Adam


# ==== 导入我们的模块 ====
# 必须先设置环境变量
os.environ["SM_FRAMEWORK"] = "tf.keras"
from models.models import ResearchModels
from utils.data_repro_pipeline import AcademicDataGenerator  # 导入新数据管道

# ------------------- GPU 与 混合精度设置 -------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 显存按需分配
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError as e:
        print(e)


# 开启混合精度
# try:
#     policy = mixed_precision.Policy('mixed_float16')
#     mixed_precision.set_global_policy(policy)
#     print(f"✅ Compute Policy: {policy.compute_dtype}")
# except Exception as e:
#     print(f"⚠️ Mixed Precision Failed: {e}")


# ------------------- 学术评估指标 (Metrics) -------------------
def precision_m(y_true, y_pred):
    # 调大 epsilon 避免除零，但在 BS=1 时不要用太大的 smooth 掩盖错误
    smooth = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 阈值二值化 (Thresholding) - 这一点很关键！
    # 之前的计算是用概率值直接乘，这叫 Soft Metric。
    # 学术评估通常用 Hard Metric (先 >0.5 变成 0/1 再算)。
    y_pred_hard = K.cast(y_pred > 0.5, tf.float32)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred_hard, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_hard, 0, 1)))

    return true_positives / (predicted_positives + smooth)


def recall_m(y_true, y_pred):
    smooth = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred_hard = K.cast(y_pred > 0.5, tf.float32)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred_hard, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    return true_positives / (possible_positives + smooth)


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + 1e-7))


def iou_coef(y_true, y_pred):
    # 这是一个 Soft IoU，用于 Loss 计算是 OK 的，但用于评估有点虚
    # 但为了保持 Loss 可微，我们这里的实现用于 Loss，
    # 我们可以单独写一个 iou_score 用于 Metrics
    smooth = 1e-7  # 大幅减小 Smooth
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection

    # 如果 Union 是 0 (即 Ground Truth 全黑 且 预测全黑)，IoU 应该是 1
    # 利用 tf.where 处理这种情况
    iou = (intersection + smooth) / (union + smooth)

    return K.mean(iou, axis=0)


# 专门用于显示的 Hard IoU (评估用)
def iou_metric(y_true, y_pred):
    y_pred_hard = K.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    intersection = K.sum(K.abs(y_true * y_pred_hard), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred_hard, [1, 2, 3]) - intersection

    smooth = 1e-7
    iou = (intersection + smooth) / (union + smooth)
    return K.mean(iou, axis=0)


# Loss 依然使用 Soft IoU 以保证梯度平滑
def iou_loss(y_true, y_pred):
    return 1.0 - iou_coef(y_true, y_pred)


# ------------------- 实验参数配置 -------------------
MODEL_NAME = 'LRDNet_Academic_Repro_v2'
# 【关键】显存修复：对于 1280x384 + VGG19，RTX 4060Ti (16GB) 建议 Batch Size = 4
# 如果依然报错 OOM，请改为 2
BATCH_SIZE = 1
WIDTH, HEIGHT = 1280, 384
EPOCHS = 1500
LR = 1e-4

print(f"🚀 Starting Academic Experiment: {MODEL_NAME}")
print(f"📏 Input Size: {WIDTH}x{HEIGHT} | Batch Size: {BATCH_SIZE}")

# ------------------- 数据管道初始化 -------------------
train_gen = AcademicDataGenerator(target='train', batch_size=BATCH_SIZE, width=WIDTH, height=HEIGHT)
val_gen = AcademicDataGenerator(target='valid', batch_size=BATCH_SIZE, width=WIDTH, height=HEIGHT)

train_data = train_gen.get_dataset()
val_data = val_gen.get_dataset()

# ------------------- 模型构建与编译 -------------------
rm = ResearchModels(modelname=MODEL_NAME, height=HEIGHT, width=WIDTH)

# 在模型编译处修改
rm.model.compile(
    optimizer=Adam(learning_rate=LR),
    loss=iou_loss,
    metrics=[iou_metric, f1_m, precision_m, recall_m]
)


# ------------------- Callbacks 设置 -------------------
checkpoints_dir = os.path.join('results', MODEL_NAME)
os.makedirs(checkpoints_dir, exist_ok=True)
model_weights_dir = os.path.join(checkpoints_dir, 'weights')
os.makedirs(model_weights_dir, exist_ok=True)

# 监控 val_f1_m (学术界更看重 F1)
ckpt_path = os.path.join(model_weights_dir, 'best_model.hdf5')
checkpointer = ModelCheckpoint(
    filepath=ckpt_path,
    verbose=1,
    save_best_only=True,
    monitor='val_f1_m',  # 保存 F1 分数最高的模型
    mode='max'
)

csv_logger = CSVLogger(os.path.join(checkpoints_dir, 'training_log.csv'))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_f1_m', mode='max', patience=50, verbose=1)

callbacks = [csv_logger, checkpointer, reduce_lr, early_stop]

# ------------------- 开始训练 -------------------
try:
    history = rm.model.fit(
        train_data,
        steps_per_epoch=len(train_gen),
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_data,
        validation_steps=len(val_gen)
    )
except KeyboardInterrupt:
    print("\n🛑 Training interrupted by user.")
except Exception as e:
    print(f"\n❌ An error occurred during training: {e}")
    print("💡 建议: 如果是 OOM 错误，请尝试将 train_repro_academic.py 中的 BATCH_SIZE 改为 2")
