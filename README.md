# Inception_V3

## InceptionV3描述

Google的InceptionV3是深度学习卷积架构系列的第3个版本。InceptionV3主要通过修改以前的Inception架构来减少计算资源的消耗。这个想法是在2015年出版的Rethinking the Inception Architecture for Computer Vision, published in 2015一文中提出的。

[论文](https://arxiv.org/pdf/1512.00567.pdf)： Min Sun, Ali Farhadi, Steve Seitz.Ranking Domain-Specific Highlights by Analyzing Edited Videos[J].2014.

## 模型架构

InceptionV3的总体网络架构如下：

[链接](https://arxiv.org/pdf/1512.00567.pdf)

## 训练过程

### 训练

```shell
  export CUDA_VISIBLE_DEVICES=0
  python train.py --model inceptionv3 --data_url ./dataset/imagenet > train.log 2>&1 &

```

```log
epoch:0 step:1251, loss is 5.7787247
Epoch time:360760.985, per step time:288.378
epoch:1 step:1251, loss is 4.392868
Epoch time:160917.911, per step time:128.631
```

## 评估过程

### 评估

使用python或shell脚本开始训练。shell脚本的用法如下：

```shell
python validate.py --model googlenet --data_url ./dataset/imagenet --checkpoint_path=[CHECKPOINT_PATH]
```

```log
metric:{'Loss':1.778, 'Top1-Acc':0.788, 'Top5-Acc':0.942}
```
