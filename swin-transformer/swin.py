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

