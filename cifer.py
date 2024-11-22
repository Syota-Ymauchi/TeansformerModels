import torch
import torch.utils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # データセットの定義
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    # DataLoader
    batch_size = 100
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    # クラスラベルの定義
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')
    images, labels = next(iter(trainloader))

    def imshow(img):
        """
        画像を表示する関数
        """
        npimg = img.numpy()
        # img [c, h, w]->[h, w, c]
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # 描画領域のサイズを設定
    fig = plt.figure(figsize=(15, 15))
    # CIFER10のデータセットの画像を10×10のグリッドで表示し
    # 各画像の下にクラスラベルを表示
    for i in range(batch_size):
        ax = fig.add_subplot(10, 10, i+1, xticks=[], yticks=[])
        imshow(images[i])
        ax.set_title(classes[labels[i]], fontsize=8, y=-0.2)
    # プロットエリアの間隔を調整
    plt.subplots_adjust(hspace=0.1, wspace=0.3)
    plt.show()
if __name__ == '__main__':
    main()
