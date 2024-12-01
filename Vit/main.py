import time

import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary

from preparation import ModelConfig, evalate
from Mymodel import VisionTransformer

def main():
    config = ModelConfig() # パラメータの初期値を設定する

    # CIFAR-10のデータセットの平均と標準偏差を用いて
    # 各チャネルのデータを正規化する変換器を作成
    normalize_transform = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # 訓練データの正規化を行う変換器
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # 水平方向にランダムに反転
        transforms.RandomCrop(config.img_size, padding=4), # ランダムに切り取る
        # 画像の明るさ、コントラスト、彩度をランダムに変更
        transforms.ColorJitter(
            brightness=0.2, # 明るさを0.8~1.2倍に変更
            contrast=0.2, # コントラスト(明るさと暗さの差)を0.8~1.2倍に変更
            saturation=0.2 # 彩度(色の鮮やかさ)を0.8~1.2倍に変更
            ),
        transforms.ToTensor(),
        normalize_transform])

    # 検証用データの正規化を行う変換器
    val_transform = transforms.Compose([transforms.ToTensor(),
                                        normalize_transform])

    # 訓練用データセットを用意
    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=train_transform)
    # 検証用データセットを用意
    val_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                train=False,
                                                download=True,
                                                transform=val_transform)
    # 訓練用データローダーを作成
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=True)
    # 検証用データローダーを作成
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=False)

    # 損失関数の定義
    loss_func = F.cross_entropy

    # Vision Transformerモデルのインスタンス化]
    model = VisionTransformer(
        len(train_dataset.classes),
        config.img_size,
        config.patch_size,
        config.num_inputlayers,
        config.num_heads,
        config.num_mlp_units,
        config.num_layers
    )

    # optimizerの定義
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=0.0005)
    # deviceの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # learning rate schedulerの定義
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=config.num_epochs)


    def print_model_summary():
        """モデルのサマリーを出力する関数
        """
        summary(model, (3, config.img_size, config.img_size))


    def train_eval():
        """学習と検証を行う関数
        """
        # グラフ用のログを初期化
        # epochごとの平均損失と平均正解率を格納するリスト
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        times = []
        best_val_loss = float('inf')

        # epochごとに学習と検証を行う
        for epoch in range(config.num_epochs):
            # 開始時間を記録
            start_time = time.time()
            # モデルを訓練モードに切り替え
            model.train()
            total_loss = 0.0
            total_accuracy = 0.0

            for x, y in tqdm(train_loader, desc='Training', total=len(train_loader), leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                preds = model(x)
                loss = loss_func(preds, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_accuracy += (preds.argmax(dim=1) == y).float().mean().item()
            
            # エポック毎の損失と正解率を計算
            train_losses.append(total_loss / len(train_loader))
            train_accuracies.append(total_accuracy / len(train_loader))

            # 検証を行う
            val_loss, val_accuracy = evalate(val_loader, model, loss_func)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            end_time = time.time()
            times.append(end_time - start_time)
            # エポック毎の損失と正解率を出力
            print(f'Epoch {epoch+1}/{config.num_epochs}')
            print(f'Time: {end_time - start_time:.2f} seconds')
            print(f'Train Loss: {train_losses[-1]:.4f}')
            print(f'Train Accuracy: {train_accuracies[-1]:.4f}')
            print(f'Val Loss: {val_losses[-1]:.4f}')
            print(f'Val Accuracy: {val_accuracies[-1]:.4f}')

            # 最良の検証損失を更新
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                # モデルの保存
                torch.save(model.state_dict(), './saved_models/best_model.pth')
                # optimizerの保存
                torch.save(optimizer.state_dict(), './saved_models/best_optimizer.pth')

            # learning rate schedulerの更新
            scheduler.step()    
        total_time = sum(times)
        print(f'Total Time: {total_time:.2f} seconds')
        # グラフの描画
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        axs[0].plot(train_losses, label='Train Loss')
        axs[0].plot(val_losses, label='Val Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[1].plot(train_accuracies, label='Train Accuracy')
        axs[1].plot(val_accuracies, label='Val Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()
        axs[2].plot(times, label='Time')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Time (seconds)')
        axs[2].legend()
        plt.tight_layout()
        plt.show()
    # 学習と検証を行う
    print_model_summary()
    train_eval()
if __name__ == '__main__':
    main()
   
