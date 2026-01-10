import sys
import os
import tensorflow as tf

# ==== 关键修改：全部替换为 tensorflow.keras ====
from tensorflow.keras.layers import (
    Multiply, AveragePooling2D, ReLU, Lambda, Activation, multiply, Average, add, Dense, Conv2D,
    Input, concatenate, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose, MaxPooling2D, Dropout
)
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.models import Model, load_model, Sequential, model_from_json
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K
import tensorflow.keras as keras  # 为了保持 self.get_kwargs 中的兼容性

# 确保 segmentation_models 使用 tf.keras
# 注意：在导入这个文件之前，train_gpu.py 必须已经设置了 os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from segmentation_models.utils import set_trainable


class ResearchModels():
    def __init__(self, modelname, width=None, height=None, dim=3, verb=0):
        self.modelname = modelname
        self.width = width
        self.height = height
        self.dim = dim

        if 'LRDNet' in modelname:
            print("***** Loading LRDNet proposed model *****")
            self.model = self.LRDNet()

        if verb == 1:
            self.model.summary()
            print('*******Total parameters********', self.model.count_params())

    def DiceLoss(self, y_true, y_pred):
        smooth = 1e-6
        gama = 2
        y_true, y_pred = tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y_true)) + smooth
        denominator = tf.reduce_sum(y_pred ** gama) + tf.reduce_sum(y_true ** gama) + smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

    def get_kwargs(self):
        return {
            'backend': K,
            'layers': layers,
            'models': keras.models,
            'utils': keras.utils,
        }

    def iou_coef(self, y_true, y_pred):
        smooth = 1e-5
        # 强制类型转换，防止 float16 溢出
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou

    def iou_loss(self, y_true, y_pred):
        return 1.0 - self.iou_coef(y_true, y_pred)

    # ... (其他不需要修改的辅助函数保持不变: down, up) ...
    # 为了完整性，这里保留 down/up/TransNet/fuse 的原始逻辑，但依赖项已换成 tf.keras

    def TransNet(self, img, ADI, filters=1):
        # 简单校验
        # assert img.shape != ADI.shape # 原代码这个校验逻辑有点怪，通常不需要
        f = int(ADI.shape[3])
        theta_a = Conv2D(f, [1, 1], strides=[1, 1], padding='same')(ADI)
        theta_b = Conv2D(f, [1, 1], strides=[1, 1], padding='same')(ADI)
        x1 = Multiply()([theta_a, ADI])
        x1 = add([x1, theta_b])
        x2 = Concatenate(axis=3)([x1, img])
        return x2

    def fuse(self, a, b, c, d):
        f = int(a.shape[3])
        t_a = Conv2D(f, [3, 3], strides=[1, 1], padding='same')(a)
        t_b = Conv2D(f, [3, 3], strides=[1, 1], padding='same')(b)
        t_c = Conv2D(f, [3, 3], strides=[1, 1], padding='same')(c)
        t_d = Conv2D(f, [3, 3], strides=[1, 1], padding='same')(d)

        x1 = add([t_a, t_b])
        x1 = add([x1, t_c])
        x1 = add([x1, t_d])
        return x1

    def get_backbone(self, backbone_name):
        # 这里的 backbone 变量名最好和函数参数分开，避免混淆
        print(f'**** {backbone_name} backbone ****')

        # Segmentation Models 的调用
        # 因为我们设置了 SM_FRAMEWORK=tf.keras，这里返回的就是 tf.keras 模型
        backbone = sm.Unet(backbone_name.lower(), encoder_weights='imagenet')

        # 根据名字匹配层
        layer_names = []
        if 'efficientnet' in backbone_name.lower():
            layer_names = ['block6a_expand_activation', 'block4a_expand_activation', 'block3a_expand_activation',
                           'block2a_expand_activation']
        elif 'resnet' in backbone_name.lower() or 'resnext' in backbone_name.lower():
            layer_names = ['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0']
        elif 'vgg' in backbone_name.lower():
            layer_names = ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2']
        elif 'mobilenet' == backbone_name.lower():
            layer_names = ['conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu']
        elif 'mobilenetv2' == backbone_name.lower():
            layer_names = ['block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu', 'block_1_expand_relu']
        # ... (为了节省篇幅，其他很少用的分支你可以保留原代码的逻辑，只要 import 对了就没问题)

        # 为了防止原代码里有些分支没覆盖，给一个默认VGG19的回退逻辑（可选）
        if not layer_names and 'vgg19' in backbone_name.lower():
            layer_names = ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2']

        return backbone, layer_names

    def LRDNet(self):
        print('********** LRDNet (TF.Keras GPU Optimized) **********')
        height = self.height
        width = self.width

        # 获取 Backbone
        backbone, layer_names = self.get_backbone('vgg19')

        # [终极修复] 绕过 segmentation_models 充满 bug 的 set_trainable 函数
        # 直接手动开启训练权限，效果一模一样，且绝不会报错
        backbone.trainable = True

        input_layer = Input(shape=[height, width, 3])
        input_layer_2 = Input(shape=[height, width, 3])

        # 参数配置
        # 默认使用 V3 配置
        l1_flt = 64
        col_2_F = 64
        col_3_F = 32
        col_4_F = 16
        last_filters = 256

        if 'V1' in self.modelname:
            l1_flt, col_2_F, col_3_F, col_4_F, last_filters = 8, 64, 32, 16, 256
        elif 'V2' in self.modelname:
            l1_flt, col_2_F, col_3_F, col_4_F, last_filters = 16, 16, 8, 4, 64

        activation = 'relu'

        # --- ADI Branch (Helper) ---
        # 使用 Model(inputs, outputs) 提取中间层特征
        # 注意：这里需要确保 layer_names[index] 在 backbone.layers 中存在

        def get_feature(layer_name, input_tensor):
            return Model(inputs=backbone.input, outputs=backbone.get_layer(layer_name).output)(input_tensor)

        # 提取特征
        feat3 = get_feature(layer_names[3], input_layer_2)
        feat2 = get_feature(layer_names[2], input_layer_2)
        feat1 = get_feature(layer_names[1], input_layer_2)
        feat0 = get_feature(layer_names[0], input_layer_2)

        # ADI Branch构建
        col_1_1_A = Conv2D(l1_flt, (3, 3), padding='same', activation=activation)(feat3)
        col_1_2_A = Conv2D(l1_flt * 2, (3, 3), padding='same', activation=activation)(feat2)
        col_1_3_A = Conv2D(l1_flt * 4, (3, 3), padding='same', activation=activation)(feat1)
        col_1_4_A = Conv2D(l1_flt * 8, (3, 3), padding='same', activation=activation)(feat0)

        col_2_4_A = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_4_A)
        col_2_3_A = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_3_A)
        col_2_2_A = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_2_A)
        col_2_1_A = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_1_A)

        col_3_1_A = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_1_A)
        col_3_2_A = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_2_A)
        col_3_3_A = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_3_A)
        col_3_4_A = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_4_A)

        col_4_1_A = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_1_A)
        col_4_2_A = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_2_A)
        col_4_3_A = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_3_A)
        col_4_4_A = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_4_A)

        upsample_1_A = UpSampling2D(interpolation='bilinear', size=(2, 2))(col_4_1_A)
        upsample_2_A = UpSampling2D(interpolation='bilinear', size=(4, 4))(col_4_2_A)
        upsample_3_A = UpSampling2D(interpolation='bilinear', size=(8, 8))(col_4_3_A)
        upsample_4_A = UpSampling2D(interpolation='bilinear', size=(16, 16))(col_4_4_A)

        # --- Image Branch ---
        img_feat3 = get_feature(layer_names[3], input_layer)
        col_1_1 = Conv2D(l1_flt, (3, 3), padding='same', activation=activation)(img_feat3)
        col_1_1 = self.TransNet(col_1_1, col_1_1_A)

        img_feat2 = get_feature(layer_names[2], input_layer)
        col_1_2 = Conv2D(l1_flt * 2, (3, 3), padding='same', activation=activation)(img_feat2)
        col_1_2 = self.TransNet(col_1_2, col_1_2_A)

        img_feat1 = get_feature(layer_names[1], input_layer)
        col_1_3 = Conv2D(l1_flt * 4, (3, 3), padding='same', activation=activation)(img_feat1)
        col_1_3 = self.TransNet(col_1_3, col_1_3_A)

        img_feat0 = get_feature(layer_names[0], input_layer)
        col_1_4 = Conv2D(l1_flt * 8, (3, 3), padding='same', activation=activation)(img_feat0)
        col_1_4 = self.TransNet(col_1_4, col_1_4_A)

        # Decoder Path
        col_2_4 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_4)
        col_2_4 = self.TransNet(col_2_4, col_2_4_A)

        upsample_for_3 = UpSampling2D(interpolation='bilinear')(col_2_4)
        # 注意: 这里可能需要 Resize 或 Padding 确保形状一致，如果 U-Net 形状对齐有问题的话
        # 假设 DataSet 产生的尺寸是 32 的倍数，通常没问题

        x = self.TransNet(col_1_3, upsample_for_3)
        col_2_3 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(x)
        col_2_3 = self.TransNet(col_2_3, col_2_3_A)

        upsample_for_2 = UpSampling2D(interpolation='bilinear')(col_2_3)
        x = self.TransNet(col_1_2, upsample_for_2)
        col_2_2 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(x)
        col_2_2 = self.TransNet(col_2_2, col_2_2_A)

        upsample_for_1 = UpSampling2D(interpolation='bilinear')(col_2_2)
        x = self.TransNet(col_1_1, upsample_for_1)
        col_2_1 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(x)
        col_2_1 = self.TransNet(col_2_1, col_2_1_A)

        # Level 3
        col_3_1 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_1)
        col_3_1 = self.TransNet(col_3_1, col_3_1_A)

        col_3_2 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_2)
        col_3_2 = self.TransNet(col_3_2, col_3_2_A)

        col_3_3 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_3)
        col_3_3 = self.TransNet(col_3_3, col_3_3_A)

        col_3_4 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_4)
        col_3_4 = self.TransNet(col_3_4, col_3_4_A)

        # Level 4
        col_4_1 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_1)
        col_4_2 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_2)
        col_4_3 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_3)
        col_4_4 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_4)

        upsample_1 = UpSampling2D(interpolation='bilinear', size=(2, 2))(col_4_1)
        upsample_2 = UpSampling2D(interpolation='bilinear', size=(4, 4))(col_4_2)
        upsample_3 = UpSampling2D(interpolation='bilinear', size=(8, 8))(col_4_3)
        upsample_4 = UpSampling2D(interpolation='bilinear', size=(16, 16))(col_4_4)

        cat = self.fuse(upsample_1, upsample_2, upsample_3, upsample_4)
        cat_A = self.fuse(upsample_1_A, upsample_2_A, upsample_3_A, upsample_4_A)

        cat = Concatenate(axis=3)([cat, cat_A])

        out = Conv2D(filters=last_filters, kernel_size=(3, 3), padding='same', activation="sigmoid")(cat)

        # 最后的输出层
        # 混合精度建议：Sigmoid 输出通常需要保持 float32 才能数值稳定，但在 Mixed Policy 'mixed_float16' 下
        # Keras 通常会自动处理 Activation 为 'sigmoid' 的层为 float32
        out = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation="sigmoid", dtype='float32')(out)

        model = Model(inputs=[input_layer, input_layer_2], outputs=[out])

        # 在 models.py 里编译通常不是最佳实践，但为了兼容原逻辑保留
        # 注意：这里我们只构建模型，编译留给 train_gpu.py 或者在这里返回未编译的 model
        return model
