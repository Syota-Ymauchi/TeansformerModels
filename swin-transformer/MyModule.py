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
            x: 入力テンソル
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
            dim(int): 線形変換後のパッチデータの次元(dim=128)
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
        # xの形状: (バッチサイズ(bs), チャネル数(3), 画像の高さ(32), 画像の幅(32))
        # F.unfoldでパッチに分割
        # kernel_size: パッチ1辺のサイズ(2)
        # stride: パッチ1辺のサイズ(2)
        #
        # F.unfoldの出力の形状: (バッチサイズ(bs), チャネル数(3), パッチ数(256), パッチのサイズ(2*2*3))
        # .movedim(-2, -1): パッチ数とチャネル数を入れ替え
        # 出力の形状: (バッチサイズ(bs), チャネル数(3), パッチ数(256), パッチのサイズ(2*2*3))
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).movedim(-2, -1)

        # パッチデータを線形変換
        # 出力の形状: (バッチサイズ(bs), パッチ数(256), パッチのサイズ(128))
        x = self.projection(x)
        # 正規化を行う
        x = self.norm(x)
        return x 

class AddPositionEmbedding(nn.Module):
    """各パッチに位置情報を追加して特徴マップを生成する

    Attributes:
        pos_embedding: 位置情報として学習可能なパラメータ
    """
    def __init__(self, dim, num_patches):
        """
        Args:
            dim(int): 線形変換後のパッチデータの次元(dim=128)
            num_patches(int): パッチの数(num_patches=256)
        """
        super().__init__()
        # (256, 128)の形状の位置情報を学習可能なパラメータを作成し、位置情報とする
        self.pos_embedding = nn.Parameter(torch.Tensor(num_patches, dim))
        
    def forward(self, x):
        """
        Args:
            x(torch.Tensor): パッチ分割後のデータ(bs, 256, 128)
        Returns:
            torch.Tensor: 位置情報を追加したテンソル
        """
        # 入力テンソルに位置情報を追加
        return x + self.pos_embedding

class ToEmbeddings(nn.Sequential):
    """レイヤーを順次実行するSequentialクラスを継承したクラスを定義
    """
    def __init__(self, in_channels, dim, patch_size, num_patches, p_drop):
        """
        Args:
            in_channels(int): 入力画像のチャネル数(in_channels=3)
            dim(int): 線形変換後のパッチデータの次元(dim=128)
            patch_size(int): パッチ1辺のサイズ(patch_size=2)
            num_patches(int): パッチの数(num_patches=256)
            p_drop(float): ドロップアウト率(p_drop=0.1)
        """
        super().__init__(
            # ToPatchesでパッチに分割
            ToPatches(in_channels, dim, patch_size),
            # AddPositionEmbeddingで位置情報を追加
            AddPositionEmbedding(dim, num_patches),
            # nn.Dropoutでドロップアウト
            nn.Dropout(p_drop)
        )

class ShiftedWindowAttention(nn.Module):
    """nn.Moduleクラスを継承したカスタムレイヤー

    Swin Transformerにおける次の機構を実装する
    . Window-based Multi-Head Self-Attention (W-MSA)
    . Shifted Window-based Multi-Head Self-Attention (SW-MSA)

    Attributes:
        heads: ヘッドの数
        head_dim: ヘッドの次元
        scale: スケールファクター
        shape: 特徴マップ256個のパッチを正方行列にした時の形状(16, 16)
        window_size: ウィンドウ1辺のサイズ(window_size=4)
        shift_size: シフトサイズ
        pos_enc: 位置情報のための学習可能なパラメータ
    """
    def __init__(self,dim, head_dim, shape, window_size, shift_size=0):
        """
        Args:
            dim(int): 特徴マップにおけるパッチのサイズ
                      dim=[128, 128, 256]から入手した第一要素の128
            head_dim(int): ヘッドの次元
            shape(tuple): 特徴マップ256個のパッチを正方行列にした時の形状(16, 16)
                          shape = (image_size // patch_size, image_size // patch_size)
                          shape = (32 // 2, 32 // 2) = (16, 16)

            window_size(int): ウィンドウ1辺のサイズ(window_size=4)
            shift_size(int): ウィンドウをシフトするサイズ(shift_size=0)
        """
        super().__init__()
        # パッチのサイズdimをヘッドのサイズhead_dimで割ってヘッドの数を計算
        self.heads = dim // head_dim
        # ヘッドのサイズをself.head_dimに設定
        self.head_dim = head_dim
        # ヘッドのサイズhead_dimの平方根の逆数を計算して
        # スケーリングファクターを求める
        # 0.5で1/2乗している
        # -をつけているのは逆数を求めるため
        self.scale = head_dim ** -0.5

        self.shape = shape # 256個のパッチを正方行列にした時の形状(16, 16)
        self.window_size = window_size # ウィンドウ1辺のサイズ(window_size=4)
        self.shift_size = shift_size # ウィンドウをシフトするサイズを設定

        # クリエ、キー、バリューを計算する全結合層を作成
        # 入力次元数: dim(128,)
        # 出力次元数: dim * 3 (128 * 3=384,)
        self.to_qkv = nn.Linear(dim, dim*3)

        # 各ヘッドの出力を結合するための全結合層を作成
        # 入力次元数: パッチサイズ(128,)
        #　ユニット数: パッチサイズ (128,)
        self.unifyheads = nn.Linear(dim, dim)
        # 相対位置エンコーディングのための学習可能なパラメータを作成
        self.pos_enc = nn.Parameter(
            torch.Tensor(self.heads, (2 * window_size - 1)**2))
        # 相対位置エンコーディングが格納された1階層テンソルをブッファに登録する
        self.register_buffer(
            "relative_position_indices",
            self.get_indices(window_size))
        # シフトサイズが0より大きい場合はマスクを適用
        if shift_size > 0:
            # self.generate_mask()でマスクをを適用
            self.register_buffer(
                "mask", 
                self.generate_mask(shape, window_size, shift_size))
            
    @staticmethod
    def get_indices(window_size):
        """相対位置インデックスを計算
        Args:
            window_size: ウィンドウ1辺のサイズ
        Returns:
            ウィンドウ内すべてのパッチを組み合わせた相対位置インデックス
            (window_size(行), window_size(列), window_size(行), window_size(列))を
            (window_size**4,)にフラット化したテンソル
        """
        # 0からwindow_size -1 までの1階層テンソル[0, 1, 2, 3]を作成
        x = torch.arange(window_size, dtype=torch.long)
        # xの要素を使って4次元のグリッドを作成
        y1, x1, y2, x2 = torch.meshgrid(x, x, x, x, indexing='ij')
        # 相対位置インデックスを求める
        indices = ((y1 - y2 + window_size - 1) * (2 * window_size - 1) + (x1 - x2 + window_size - 1)).flatten()
        return indices
        









