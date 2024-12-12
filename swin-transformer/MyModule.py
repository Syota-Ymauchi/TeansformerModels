import torch.nn as nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """nn.Moduleクラスを継承した残差接続を行うレイヤーを定義

    Attributes:
        residual:
            受け取ったレイヤーをnn.Sequentialで連結したシーケンシャルモデル
        gamma:
            学習可能なパラメータ
    """
    def __init__(self, *layers):
        """
        Args:
            *layers:(可変長引数): 残差ブロック内で使用されるレイヤー
        """
        super().__init__()
        # 受け取ったレイヤーをnn.Sequentialで連結したシーケンシャルモデルを作成
        self.residual = nn.Sequential(*layers)
        # 学習可能な初期値0のパラメータを作成
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        Args:
            x:
                入力テンソル
        Returns:
            入力テンソルに残差ブロックの出力を加算したテンソル
        """
        return x + self.gamma * self.residual(x)
    

class GlobalAveragePool(nn.Module):
    """GAP層を定義
    """
    def forward(self, x):
        """順伝播でGAP適用

        入力テンソルxの末尾から2番目の次元(パッチ数)について、バッチ毎のチャネル次元の平均を計算して返す
        """
        return x.mean(dim=-2)


class ToPatches(nn.Module):
    """1枚の画像を小さなパッチに分割する

    Attributes:
        patch_size: パッチ1辺のサイズ(2)
        projection: パッチデータを線形変換する全結合層
    """
    def __init__(self, in_channels, dim, patch_size):
        """
        Args:
            in_channels(int): 入力画像のチャネル数(in_channels=3)
            dim(int): 先見変換後のパッチデータの次元(dim=128)
            patch_size(int): パッチ1辺のサイズ(patch_size=2)
        """
        super().__init__()
        self.patch_size = patch_size
        # チャネル数にパッチ1辺のサイズの2乗を掛けてパッチ1個のサイズを計算
        # 3 * 2 * 2 = 12
        patch_dim = in_channels * patch_size * patch_size
        # 入力サイズをパッチのサイズ(12)
        # ユニット数をdim(128)に設定した全結合層を作成
        self.projection = nn.Linear(patch_dim, dim)
        # 正規化を行う
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """パッチに分割する一連の順伝播処理
        Args:
            x(torch.Tensor): 入力画像
        Returns:
            torch.Tensor: パッチに分割されたテンソル
        """
        # F.unfoldでパッチに分割
        # kernel_size: パッチ1辺のサイズ(2)
        # stride: パッチ1辺のサイズ(2)
        #
        # F.unfoldの出力の形状: (バッチサイズ(bs), チャネル数(3), パッチ数(256), パッチのサイズ(2*2*3))
        # .movedim(-2, -1): パッチ数とチャネル数を入れ替え
        # 出力の形状: (バッチサイズ(bs), チャネル数(3), パッチ数(256), パッチのサイズ(2*2*3))
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).movedim(-2, -1)

        # パッチデータを線形変換
        x = self.projection(x)
        # 正規化を行う
        x = self.norm(x)
        return x 
       

    


