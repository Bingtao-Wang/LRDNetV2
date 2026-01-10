import os
import sys

# 【重要】配置 segmentation_models 使用 tensorflow.keras
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D, AveragePooling2D,
    UpSampling2D, Concatenate, Multiply, Add, Activation, Lambda,
    Dropout, Dense, GlobalAveragePooling2D, add, multiply
)
from tensorflow.keras.optimizers import Adam

# 确保已安装 segmentation-models
import segmentation_models as sm
from segmentation_models.utils import set_trainable


class ResearchModels():
    def __init__(self, modelname, width=None, height=None, dim=3, verb=0):
        self.modelname = modelname
        self.width = width
        self.height = height
        self.dim = dim
        self.model = None

        if 'LRDNet' in modelname:
            print("***** Loading LRDNet proposed model *****")
            self.model = self.LRDNet()
        else:
            print(f"Warning: Model name '{modelname}' does not contain 'LRDNet'.")

        if verb == 1 and self.model is not None:
            self.model.summary()
            print('******* Total parameters ********', self.model.count_params())

    # ================= Metrics & Loss (修复 NaN 关键部分) =================
    def dice_coef(self, y_true, y_pred):
        # 【关键修改】强制转换为 float32，防止混合精度下的数值下溢
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        smooth = 1e-5
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)

    def iou_coef(self, y_true, y_pred):
        # 【关键修改】强制转换为 float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        smooth = 1e-5
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou

    def iou_loss(self, y_true, y_pred):
        return 1.0 - self.iou_coef(y_true, y_pred)

    # ================= Helper Blocks =================
    def TransNet(self, img, ADI):
        # 简单的特征融合模块
        f = int(ADI.shape[-1])
        theta_a = Conv2D(f, [1, 1], strides=[1, 1], padding='same')(ADI)
        theta_b = Conv2D(f, [1, 1], strides=[1, 1], padding='same')(ADI)

        x1 = Multiply()([theta_a, ADI])
        x1 = Add()([x1, theta_b])
        x2 = Concatenate(axis=3)([x1, img])
        return x2

    def fuse(self, a, b, c, d):
        # 多尺度融合
        f = int(a.shape[-1])
        t_a = Conv2D(f, [3, 3], padding='same')(a)
        t_b = Conv2D(f, [3, 3], padding='same')(b)
        t_c = Conv2D(f, [3, 3], padding='same')(c)
        t_d = Conv2D(f, [3, 3], padding='same')(d)

        x = Add()([t_a, t_b])
        x = Add()([x, t_c])
        x = Add()([x, t_d])
        return x

    def get_backbone(self, backbone_name):
        print(f'**** Initializing backbone: {backbone_name} ****')
        # 使用 segmentation_models 加载预训练权重
        backbone = sm.Unet(backbone_name, encoder_weights='imagenet')

        # 定义需要提取特征的层名称 (针对 VGG19)
        if backbone_name == 'vgg19':
            layer_names = ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2']
        else:
            layer_names = ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2']

        return backbone, layer_names

    # ================= Main Model Architecture =================
    def LRDNet(self):
        height = self.height
        width = self.width

        # 1. 获取主干网络
        backbone, layer_names = self.get_backbone('vgg19')

        # 2. 冻结或微调设置 (此处设为可训练)
        backbone.trainable = True

        # 3. 定义输入
        input_layer = Input(shape=(height, width, 3), name='input_img')
        input_layer_2 = Input(shape=(height, width, 3), name='input_adi')

        # 4. 配置滤波器参数 (默认为 V3 配置)
        l1_flt = 64
        col_2_F = 64
        col_3_F = 32
        col_4_F = 16
        last_filters = 256
        activation = 'relu'

        # 辅助函数：从 backbone 提取特定层输出
        def get_feat(inp, layer_name):
            return Model(inputs=backbone.input, outputs=backbone.get_layer(layer_name).output)(inp)

        # ---------------- ADI Branch (辅助输入分支) ----------------
        # 提取不同尺度的特征
        x_adi_0 = get_feat(input_layer_2, layer_names[0])  # block5
        x_adi_1 = get_feat(input_layer_2, layer_names[1])  # block4
        x_adi_2 = get_feat(input_layer_2, layer_names[2])  # block3
        x_adi_3 = get_feat(input_layer_2, layer_names[3])  # block2

        # 对应原代码 col_1_x_A
        feat_A_1 = Conv2D(l1_flt, (3, 3), padding='same', activation=activation)(x_adi_3)
        feat_A_2 = Conv2D(l1_flt * 2, (3, 3), padding='same', activation=activation)(x_adi_2)
        feat_A_3 = Conv2D(l1_flt * 4, (3, 3), padding='same', activation=activation)(x_adi_1)
        feat_A_4 = Conv2D(l1_flt * 8, (3, 3), padding='same', activation=activation)(x_adi_0)

        # 对应原代码 col_2_x_A
        feat_A_2_1 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(feat_A_1)
        feat_A_2_2 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(feat_A_2)
        feat_A_2_3 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(feat_A_3)
        feat_A_2_4 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(feat_A_4)

        # 对应原代码 col_3_x_A
        feat_A_3_1 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(feat_A_2_1)
        feat_A_3_2 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(feat_A_2_2)
        feat_A_3_3 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(feat_A_2_3)
        feat_A_3_4 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(feat_A_2_4)

        # 对应原代码 col_4_x_A
        feat_A_4_1 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(feat_A_3_1)
        feat_A_4_2 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(feat_A_3_2)
        feat_A_4_3 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(feat_A_3_3)
        feat_A_4_4 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(feat_A_3_4)

        # 上采样对齐
        up_A_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(feat_A_4_1)
        up_A_2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(feat_A_4_2)
        up_A_3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(feat_A_4_3)
        up_A_4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(feat_A_4_4)

        # 融合 ADI 特征
        cat_A = self.fuse(up_A_1, up_A_2, up_A_3, up_A_4)

        # ---------------- Image Branch (主图像分支) ----------------
        x_img_0 = get_feat(input_layer, layer_names[0])
        x_img_1 = get_feat(input_layer, layer_names[1])
        x_img_2 = get_feat(input_layer, layer_names[2])
        x_img_3 = get_feat(input_layer, layer_names[3])

        # Stage 1: Initial Conv + TransNet Fusion
        feat_1_1 = Conv2D(l1_flt, (3, 3), padding='same', activation=activation)(x_img_3)
        feat_1_1 = self.TransNet(feat_1_1, feat_A_1)

        feat_1_2 = Conv2D(l1_flt * 2, (3, 3), padding='same', activation=activation)(x_img_2)
        feat_1_2 = self.TransNet(feat_1_2, feat_A_2)

        feat_1_3 = Conv2D(l1_flt * 4, (3, 3), padding='same', activation=activation)(x_img_1)
        feat_1_3 = self.TransNet(feat_1_3, feat_A_3)

        feat_1_4 = Conv2D(l1_flt * 8, (3, 3), padding='same', activation=activation)(x_img_0)
        feat_1_4 = self.TransNet(feat_1_4, feat_A_4)

        # Stage 2: Decoder Pathway
        # Path 4 -> 3
        feat_2_4 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(feat_1_4)
        feat_2_4 = self.TransNet(feat_2_4, feat_A_2_4)

        up_2_4 = UpSampling2D(interpolation='bilinear')(feat_2_4)
        x = self.TransNet(feat_1_3, up_2_4)  # Skip connection logic

        feat_2_3 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(x)
        feat_2_3 = self.TransNet(feat_2_3, feat_A_2_3)

        # Path 3 -> 2
        up_2_3 = UpSampling2D(interpolation='bilinear')(feat_2_3)
        x = self.TransNet(feat_1_2, up_2_3)

        feat_2_2 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(x)
        feat_2_2 = self.TransNet(feat_2_2, feat_A_2_2)

        # Path 2 -> 1
        up_2_2 = UpSampling2D(interpolation='bilinear')(feat_2_2)
        x = self.TransNet(feat_1_1, up_2_2)

        feat_2_1 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(x)
        feat_2_1 = self.TransNet(feat_2_1, feat_A_2_1)

        # Stage 3
        feat_3_1 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(feat_2_1)
        feat_3_1 = self.TransNet(feat_3_1, feat_A_3_1)

        feat_3_2 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(feat_2_2)
        feat_3_2 = self.TransNet(feat_3_2, feat_A_3_2)

        feat_3_3 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(feat_2_3)
        feat_3_3 = self.TransNet(feat_3_3, feat_A_3_3)

        feat_3_4 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(feat_2_4)
        feat_3_4 = self.TransNet(feat_3_4, feat_A_3_4)

        # Stage 4
        feat_4_1 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(feat_3_1)
        feat_4_2 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(feat_3_2)
        feat_4_3 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(feat_3_3)
        feat_4_4 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(feat_3_4)

        #  Upsampling
        up_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(feat_4_1)
        up_2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(feat_4_2)
        up_3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(feat_4_3)
        up_4 = UpSampling2D(size=(16, 16), interpolation='bilinear')(feat_4_4)

        cat = self.fuse(up_1, up_2, up_3, up_4)

        # ----------------  Head (Output) ----------------
        # 融合两路特征
        final_cat = Concatenate(axis=3)([cat, cat_A])

        # 1. 倒数第二层改用 relu 激活，更稳定
        x = Conv2D(filters=last_filters, kernel_size=(3, 3), padding='same', activation="relu")(final_cat)

        # 2. 【关键修改】最后一层必须指定 dtype='float32'
        # 确保输出概率值是 FP32 精度，防止 Loss 计算出现 NaN
        output_layer = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation="sigmoid", dtype='float32')(x)

        model = Model(inputs=[input_layer, input_layer_2], outputs=[output_layer])

        # 编译模型
        optimizer = Adam(learning_rate=5e-6)  # 注意：TF2 中参数名是 learning_rate
        metrics = [self.iou_coef, self.dice_coef]
        model.compile(optimizer=optimizer, loss=self.iou_loss, metrics=metrics)

        return model
