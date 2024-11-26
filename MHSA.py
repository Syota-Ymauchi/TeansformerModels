import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """Multi-Hrad self-Attention(マルチヘッド自己注意機構)
    Attribute:
        num_heads: マルチヘッドの数
        expansion_layer: 特徴マップの数を×3するための全結合層
        headjoin_layer: 各ヘッドから出力された特徴表現を線形変換するための全結合層
        scal: ソフトマックス関数入力前に適用するスケール値
    """
    def __init__(self, num_inputlayer_units: int, num_heads: int):
        super().__init__()
        """マルチヘッドアテンションに必要なレイヤー等の定義
        Args:
            num_inputlayer_units(int): 全結合層のユニット数
            num_heads(int): マルチヘッドアテンションのヘッド数
        """
        # 特徴マップの特徴量をヘッドの数で分割出来るか確認
        if num_inputlayer_units % num_heads != 0:
            raise ValueError('num_inpitlayer_units must be divisible by num_heads')
        # ヘッドの数の指定
        self.num_heads = num_heads
        # 特徴マップ生成機構の全結合層ユニットをヘッドの数で割ることで
        # ヘッド毎の特徴量の次元を求める(書籍でいう128次元)
        dim_head = num_inputlayer_units // num_heads

        # データ拡張を行う全結合層の定義
        # 入力次元: 特徴マップの特徴量次元
        # 出力次元(ユニット数):　特徴量次元×3
        self.expansion_layer = nn.Linear(num_inputlayer_units, num_inputlayer_units*3)

        # ソフトマックス関数のオーバーフロー対策のためのスケール値
        # 次元数の平方根(1/sart(dimention))
        self.scale = 1 / (dim_head ** 0.5)

        # 各ヘッドからの出力を線形変換する全結合層の定義
        # 入力の次元数: ヘッドごとの特徴量次元=特徴マップ生成機構の全結合層ユニット数
        # 出力の次元数 : 入力の次元数と同じ
        self.headjoin_layer = nn.Linear(num_inputlayer_units, num_inputlayer_units)

        def forward(self, x: torch.Tensor):
            """順伝播の処理を行う
                Args: 
                    x(torch.Tensor) : 特徴マップ(batch size, 特徴量数
                                      (1つ目のMHSAならクラストークン数+パッチサイズ数), 特徴量次元)
                    
            """
            # 入力する特徴マップのテンソル(bs, 5, 512)から
            # バッチサイズと特徴量数を取得
            bs, ns = x.shape[:2]

            # 全結合層expantion_layerに入力してデータを拡張
            # 入力: (バッチサイズ, 5, 512)
            # 出力: (バッチサイズ, 5, 1536[512×3])
            qkv = self.expantion_layer(x)

            # view()の処理
            # データ拡張したテンソル(バッチサイズ, 5, 1536)を
            # クリエ行列、キー行列、バリュー行列に分割 -> (バッチサイズ, 5, 3, 512)
            # さらにマルチヘッドに分割 -> (バッチサイズ, 5, 3, 128)
            # .permuteの処理
            # クリエ、キー、バリューの次元をテンソルの先頭に移動して
            # (3, バッチサイズ, ヘッド数, 特長量数, 特徴量数の次元)の形状にする
            # これにより、クリエ、キー、バリューが別々の次元に配置される
            # -> [3, バッチサイズ, 4, 5, 128]
            qkv = qkv.view(bs, ns, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)       