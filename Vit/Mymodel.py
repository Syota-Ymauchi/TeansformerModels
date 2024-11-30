import torch
import torch.nn as nn

from MyModule import EncoderBlock

class VisionTransformer(nn.Module):
    """
    Atributes:
        img_size: 入力画像の縦横のサイズ
        patch_size: パッチの一辺のサイズ
        input_layer: パッチデータを線形変換するMLP
        pos_embed: 位置情報埋め込み
        class_token: クラス埋め込み
        encoder_layer: Encoderブロック
        normalize: 分類器の正規化層
        output_layer: 分類器の全結合層
    """
    def __init__(self,
                  num_classes: int,
                  img_size: int,
                  patch_size: int,
                  num_inputlayer_units: int,
                  num_heads: int,
                  num_mlp_units: int,
                  num_layers: int):
        """
        Atributes:
            num_classes(int): 画像分類のクラス数
            img_size(int): 入力画像の1辺のサイズ
            patch_size(int): パッチの一辺のサイズ
            num_inputlayer_units(int): 全結合層のユニットの数
            num_heads(int): マルチヘッドアテンションのヘッド数
            num_mlp_units(int): 多層パーセプトロンのユニット数
            num_layers(int): Encoderブロックの層数
        """
        super().__init__()
        # ------[1]パッチへの分割とそれぞれのパッチのデータをフラット化して線形変換------
        self.img_size = img_size
        self.patch_size = patch_size
        # 分割されるパッチの数
        num_patches = (img_size // patch_size) ** 2
        # 各パッチのピクセルデータ数(chanell * patch_size^2)
        input_dim = 3 * patch_size ** 2
        # パッチデータを線形変換するMLP
        self.input_layer = nn.Linear(input_dim, num_inputlayer_units)
        # ------[2]クラストークン、位置情報埋め込み------
        # クラストークンの定義
        # [1, 1, 512] を作成して標準正規分布からランダムにサンプリングして初期化
        # nn.Parameter()で学習可能なパラメータとして定義する
        self.class_token = nn.Parameter(
            torch.zeros(1, 1, num_inputlayer_units)
        )
        # 位置情報の定義
        # [1, 5(パッチ数+クラストークン分の1), 512] を作成して標準正規分布からランダムにサンプリングして初期化
        # nn.Parameter()で学習可能なパラメータとして定義する
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, num_inputlayer_units)
        )
        # ------[3]Encoderブロックの定義------
        # Encoderブロック
        self.encoder_layer = nn.ModuleList([
            EncoderBlock(num_inputlayer_units, num_heads, num_mlp_units)
            for _ in range(num_layers)
        ])
        # ------分類器------
        # 正規化層の入力次元は特徴量マップ生成機構の全結合層のユニット数
        self.normalize = nn.LayerNorm(num_inputlayer_units)
        # 正規化層の入力をnum_classesのユニットで処理
        self.output_layer = nn.Linear(num_inputlayer_units, num_classes)
        

    def forward(self, x: torch.Tensor):
        """
        Args:
            x(torch.Tensor): input data shapeは[32, 3, 32, 32]
        """
        # ------特徴マップ生成機構の適用------
        bs, c, h, w = x.shape

        # 1毎の画像を4毎のパッチに分割する
        # 今回はパッチサイズを16にした前程
        # バッチサイズは32にして前程
        # [32, 3, 32, 32] -> [32, 3, 2, 16, 2, 16]
        x = x.view(bs, c, h//self.patch_size, self.patch_size, \
                    w//self.patch_size, self.patch_size)
        # パッチ毎にフラット化
        # [32, 3, 2, 16, 2, 16] -> [32, 2, 2, 3 16, 16]
        x = x.permute(0, 2, 4, 1, 3, 5) 
        # [32, 2, 2, 16, 16] -> [32, 4, 16*16*3]
        x = x.reshape(bs, (h//self.patch_size)*(w//self.patch_size), -1)

        # 全結合層に768次元のxを入力して512次元に線形
        # 次元削減、パラメータ削減の効果
        # 局所的な情報の集約
        # 汎化性能の向上
        x = self.input_layer(x) # [32, 4, 768] -> [32, 4, 512](512はコンストラクタで指定)

        # クラストークンのテンソルをミニバッチの数だけ作成
        class_token = self.class_token.expand(bs, -1, -1) # [1, 1, 512] -> [32, 1, 512]
        # クラストークンを全結合層からの出力のdim=1の先頭に結合する
        # x : [32, 4, 512] -> [32, 5, 512]
        x = torch.cat((class_token, x), dim=1)
        
        # 学習可能な位置情報を加算
        x += self.pos_embed # [32, 5, 512] (加算なのでサイズは変わらない)

        # EncoderBlockで処理
        for layer in self.encoder_layer:
            x = layer(x)
        # Encoder処理後の特徴マップからクラストークンを抽出
        x = x[:, 0]
        # 正規化層に入力
        x = self.normalize(x)
        # 最終層に入力
        x = self.output_layer(x)
        # 戻り値は要素数num_classesのベクトル
        return x

    def get_device(self):
        """
        モデルのデバイスを取得する(最終層のパラメータのデバイスを取得)
        """
        return self.output_layer.weight.device
        


    
        