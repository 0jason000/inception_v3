from functools import partial

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierUniform

from .utils import load_pretrained
from .registry import register_model
from .layers.pooling import GlobalAvgPooling
from .layers.conv_norm_act import Conv2dNormActivation
from mission.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT


'''
1. 基本结构和timm都是一样的，没有太大差别。
2. 同样将 conv2d+bn+relu 改成了Conv2dNormActivation
3. 初始化权重后期统一整改。
4. 全局平均池化使用提出来公用的GlobalAvgPooling
'''


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'dataset_transform': {
            'transforms_imagenet_train': {
                'image_resize': 224,
                'scale': (0.08, 1.0),
                'ratio': (0.75, 1.333),
                'hflip': 0.5,
                'interpolation': 'bilinear',
                'mean': IMAGENET_DEFAULT_MEAN,
                'std': IMAGENET_DEFAULT_STD,
            },
            'transforms_imagenet_eval': {
                'image_resize': 224,
                'crop_pct': DEFAULT_CROP_PCT,
                'interpolation': 'bilinear',
                'mean': IMAGENET_DEFAULT_MEAN,
                'std': IMAGENET_DEFAULT_STD,
            },
        },
        'first_conv': 'conv1', 'classifier': 'fc',
        **kwargs
    }


default_cfgs = {
    'inception_v3': _cfg(url='')
}

weight = XavierUniform()
norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.9997)
Conv2dNormActivation = partial(Conv2dNormActivation, norm=norm, weight=weight)


class InceptionA(nn.Cell):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.concat = ops.Concat(axis=1)
        self.branch0 = Conv2dNormActivation(in_channels, 64, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, 48, kernel_size=1),
            Conv2dNormActivation(48, 64, kernel_size=5)
        ])
        self.branch2 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, 64, kernel_size=1),
            Conv2dNormActivation(64, 96, kernel_size=3),
            Conv2dNormActivation(96, 96, kernel_size=3)

        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            Conv2dNormActivation(in_channels, pool_features, kernel_size=1)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, x2, branch_pool))
        return out


class InceptionB(nn.Cell):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.concat = ops.Concat(axis=1)
        self.branch0 = Conv2dNormActivation(in_channels, 384, kernel_size=3, stride=2, pad_mode='valid')
        self.branch1 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, 64, kernel_size=1),
            Conv2dNormActivation(64, 96, kernel_size=3),
            Conv2dNormActivation(96, 96, kernel_size=3, stride=2, pad_mode='valid')

        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, branch_pool))
        return out


class InceptionC(nn.Cell):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.concat = ops.Concat(axis=1)
        self.branch0 = Conv2dNormActivation(in_channels, 192, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, channels_7x7, kernel_size=1),
            Conv2dNormActivation(channels_7x7, channels_7x7, kernel_size=(1, 7)),
            Conv2dNormActivation(channels_7x7, 192, kernel_size=(7, 1))
        ])
        self.branch2 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, channels_7x7, kernel_size=1),
            Conv2dNormActivation(channels_7x7, channels_7x7, kernel_size=(7, 1)),
            Conv2dNormActivation(channels_7x7, channels_7x7, kernel_size=(1, 7)),
            Conv2dNormActivation(channels_7x7, channels_7x7, kernel_size=(7, 1)),
            Conv2dNormActivation(channels_7x7, 192, kernel_size=(1, 7))
        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            Conv2dNormActivation(in_channels, 192, kernel_size=1)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, x2, branch_pool))
        return out


class InceptionD(nn.Cell):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.concat = ops.Concat(axis=1)
        self.branch0 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, 192, kernel_size=1),
            Conv2dNormActivation(192, 320, kernel_size=3, stride=2, pad_mode='valid')
        ])
        self.branch1 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, 192, kernel_size=1),
            Conv2dNormActivation(192, 192, kernel_size=(1, 7)),  # check
            Conv2dNormActivation(192, 192, kernel_size=(7, 1)),
            Conv2dNormActivation(192, 192, kernel_size=3, stride=2, pad_mode='valid')
        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, branch_pool))
        return out


class InceptionE(nn.Cell):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.concat = ops.Concat(axis=1)
        self.branch0 = Conv2dNormActivation(in_channels, 320, kernel_size=1)
        self.branch1 = Conv2dNormActivation(in_channels, 384, kernel_size=1)
        self.branch1_a = Conv2dNormActivation(384, 384, kernel_size=(1, 3))
        self.branch1_b = Conv2dNormActivation(384, 384, kernel_size=(3, 1))
        self.branch2 = nn.SequentialCell([
            Conv2dNormActivation(in_channels, 448, kernel_size=1),
            Conv2dNormActivation(448, 384, kernel_size=3)
        ])
        self.branch2_a = Conv2dNormActivation(384, 384, kernel_size=(1, 3))
        self.branch2_b = Conv2dNormActivation(384, 384, kernel_size=(3, 1))
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode='same'),
            Conv2dNormActivation(in_channels, 192, kernel_size=1)
        ])

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = self.concat((self.branch1_a(x1), self.branch1_b(x1)))
        x2 = self.branch2(x)
        x2 = self.concat((self.branch2_a(x2), self.branch2_b(x2)))
        branch_pool = self.branch_pool(x)
        out = self.concat((x0, x1, x2, branch_pool))
        return out


class InceptionAux(nn.Cell):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.avg_pool = nn.AvgPool2d(5, stride=3, pad_mode='valid')
        self.conv2d_0 = Conv2dNormActivation(in_channels, 128, kernel_size=1)
        self.conv2d_1 = Conv2dNormActivation(128, 768, kernel_size=5, pad_mode='valid')
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(in_channels, num_classes)

    def construct(self, x):
        x = self.avg_pool(x)
        x = self.conv2d_0(x)
        x = self.conv2d_1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class InceptionV3(nn.Cell):
    def __init__(self, num_classes=1000, aux_logits=True, in_channels=3, dropout_rate=0.8):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.Conv2d_1a = Conv2dNormActivation(in_channels, 32, kernel_size=3, stride=2, pad_mode='valid')
        self.Conv2d_2a = Conv2dNormActivation(32, 32, kernel_size=3, stride=1, pad_mode='valid')
        self.Conv2d_2b = Conv2dNormActivation(32, 64, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b = Conv2dNormActivation(64, 80, kernel_size=1)
        self.Conv2d_4a = Conv2dNormActivation(80, 192, kernel_size=3, pad_mode='valid')
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.pool = GlobalAvgPooling(keep_dims=True)
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)
        self.flatten = nn.Flatten()
        self.num_features = 2048
        self.classifier = nn.Dense(self.num_features, num_classes)

    def construct_preaux(self, x):
        x = self.Conv2d_1a(x)
        x = self.Conv2d_2a(x)
        x = self.Conv2d_2b(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b(x)
        x = self.Conv2d_4a(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        return x

    def construct_postaux(self, x):
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x

    def get_features(self, x):
        x = self.construct_preaux(x)
        x = self.construct_postaux(x)
        return x

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes):
        self.classifier = nn.Dense(self.num_features, num_classes)

    def construct(self, x):
        x = self.construct_preaux(x)

        if self.training and self.aux_logits:
            aux_logits = self.AuxLogits(x)
        else:
            aux_logits = None

        x = self.construct_postaux(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x, aux_logits


@register_model
def inception_v3(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['inception_v3']
    model = InceptionV3(num_classes=num_classes, aux_logits=True, in_channels=in_channels, **kwargs)
    model.dataset_transform = default_cfg['dataset_transform']

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)

    return model
