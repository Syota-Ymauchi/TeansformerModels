import numpy as np
from collections import defaultdict # デフォルト値を設定できる辞書
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 以下、Pytorn-Igniteからのインポート
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator # エンジン
import ignite.metrics as metrics # メトリクス
import ignite.contrib.handlers as handlers # ハンドラ


# パラメータの値の設定
DATA_DIR = "./data"
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_WORKERS = 2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-1

# 使用可能なデバイスの取得
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Augmentation
train_transform = transforms.Compose([
    # 画像をランダムに左右反転(p=0.5)
    transforms.RandomHorizontalFlip(p=0.5),
    # 画像の各辺に4ピクセルのパディングを追加して画像をランダムに抜き取る
    transforms.RandomCrop(size=(IMAGE_SIZE, IMAGE_SIZE), padding=4),
    # PIL画像をtorch.Tensorに変換
    transforms.ToTensor(),
    # 画像のデータ型をtorch.floatに変換
    transforms.ConvertImageDtype(torch.float)
])

# データセットの読み込み
train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True,\
                                 transform=train_transform)
test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True,\
                               transform=transforms.ToTensor())

# データローダーの作成
# GPUが利用可能な場合のみTrue、デバイスのメモリをピンポイントでピンメモリにする
pin_memory = torch.cuda.is_available()  
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE,    
    shuffle=False, num_workers=NUM_WORKERS,
    pin_memory=pin_memory)
