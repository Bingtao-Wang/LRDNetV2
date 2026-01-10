import os
import time
import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras import mixed_precision # 【新增 1】

# ==================== 1. 环境与显卡配置 ====================
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 显存按需分配
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 显存增长模式已开启: {gpus}")
    except RuntimeError as e:
        print(f"❌ GPU 配置错误: {e}")

# 【新增 2】开启混合精度 (FP16) - 省显存神器
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"⚡ 混合精度训练已开启: {policy.name}")

# 忽略警告
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
# ==========================================================

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# 导入本地模块
# 确保 E:\GitHub\LRDNet 在 PYTHONPATH 中，或者直接在根目录运行此脚本
from models.models import ResearchModels
from utils.data import DataSet
from utils.data_aug import DataSet_aug

# 清屏 (可选)
# os.system('cls')

# ==================== 2. 参数设置 ====================
model_name = 'LRDNet_Test_SM——'
augmentation = False
save_best_only = True
seeding = False
# 建议：RTX 4060 Ti 16G 可以尝试 batch_size = 4 或 8，速度会快很多
batch_size = 2
patience = 15
epochs = 1500
save_models = True

if seeding:
    from numpy.random import seed
    seedi = 100
    seed(seedi)
    tf.random.set_seed(seedi)

# 根据模型名称确定输入尺寸
if 'LRDNet' in model_name:
    width = 1280
    height = 384
    print('************** Using Size 1280 x 384 **************')
elif 'SM' in model_name:
    width = 256
    height = 256
    print('************** Using Size 256 x 256 **************')
else:
    # 默认值
    width = 256
    height = 256

# ==================== 3. 数据加载 ====================
if augmentation:
    print(">>> 使用数据增强模式")
    train_images = DataSet_aug(model=model_name, target='train', batch_size=batch_size, width=width, height=height)
    val_images = DataSet_aug(model=model_name, target='valid', batch_size=batch_size, width=width, height=height)
    aug_suffix = '[AUGBIG]'
else:
    print(">>> 使用普通数据模式")
    train_images = DataSet(model=model_name, target='train', batch_size=batch_size, width=width, height=height)
    val_images = DataSet(model=model_name, target='valid', batch_size=batch_size, width=width, height=height)
    aug_suffix = ''

steps_per_epoch = train_images.steps_per_epoch
validation_steps = val_images.validation_steps
train_data = train_images.td
val_data = val_images.vd

# ==================== 4. 路径与回调函数 ====================
# 结果保存路径
checkpoints_dir = os.path.join('results', model_name)
model_weights_dir = os.path.join(checkpoints_dir, model_name + '_Weights')
logs_dir = os.path.join(checkpoints_dir, model_name + '_logs')

# 自动创建文件夹
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(model_weights_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# 1. 模型保存 (Checkpointer)
# 修正了文件名格式，使其兼容 Windows 路径
filepath = os.path.join(model_weights_dir, model_name + aug_suffix + '.({epoch:03d})-[{loss:.3f}]-[{val_loss:.3f}].hdf5')
checkpointer = ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    save_best_only=save_best_only,
    monitor='val_loss', # 或者是 'val_iou_coef'，根据你的需求
    mode='auto'
)

# 2. TensorBoard 日志
tb = TensorBoard(log_dir=os.path.join(logs_dir, 'tensorboard'))

# 3. CSV 日志
timestamp = str(int(time.time()))
csv_logger = CSVLogger(os.path.join(logs_dir, f'{model_name}_{timestamp}.log'))

# 4. 学习率调整
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# 5. 早停
early_stopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

# 组合回调函数
if save_models:
    callbacks_list = [tb, csv_logger, checkpointer, reduce_lr, early_stopper]
else:
    callbacks_list = [tb, csv_logger, reduce_lr, early_stopper]

# ==================== 5. 模型构建与训练 ====================
print(f">>> 正在构建模型: {model_name}")
rm = ResearchModels(modelname=model_name, height=height, width=width)

print(">>> 开始训练...")
# 使用 .fit() 替代旧版 .fit_generator()
history = rm.model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks_list,
    validation_data=val_data,
    validation_steps=validation_steps
)

print(">>> 训练结束")
