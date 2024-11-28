import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """Multi-Hrad self-Attention(マルチヘッド自己注意機構)
    Attributes:
        num_heads: マルチヘッドの数
        expansion_layer: 特徴マップの数を×3するための全結合層
        headjoin_layer: 各ヘッドから出力された特徴表現を線形変換するための全結合層
        scal: ソフトマックス関数入力前に適用するスケール値
    """
    def __init__(self, num_inputlayer_units: int, num_heads: int):
        """マルチヘッドアテンションに必要なレイヤー等の定義
        Args:
            num_inputlayer_units(int): 全結合層のユニット数
            num_heads(int): マルチヘッドアテンションのヘッド数
        """
        super().__init__()
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

        # クリエ行列(q), キー行列(k), バリュー行列(v)に分割
        # q, k, vそれぞれが[バッチサイズ, 4, 5, 128]となる
        # それぞれヘッドで分割されている
        q, k, v = qkv.unbind(dim=0)

        # それぞれのヘッドごとのクリエ行列(5, 128)と転置したキー行列(128, 5)の行列を計算し、
        # 各要素間の関連度(アテンションスコア)を求める、結果のテンソルの形状は(5, 5)
        attn = q @ k.transpose(-2, -1)
        attn = F.softmax(attn*self.scale, dim=-1)

        # アテンションスコアattnとバリュー行列で行列積を計算
        # attn : [bs, 4, 5, 5] @ value : [bs, 4, 5, 128] ->
        # [bs, 4, 5, 128]
        x = attn @ v 

        # permuteの処理
        # バッチサイズ(32), ヘッド数(4), 特徴量数(5), 特徴量次元(128)を
        # バッチサイズ(32), 特徴量数, ヘッド数, 特徴量次元に並び替える
        # dim=2でflatten -> [バッチサイズ(32), 5, 4×3] 
        x = x.permute(0, 2, 1, 3).flatten(2)

        # 全結合層headjoin_layerに入力
        # 入力 x : [32, 5, 512]
        # 出力 x : [32, 5, 512]
        x = self.headjoin_layer(x)

        return x


class MLP(nn.Module):
    """多層パーセプトロンの定義
    Transformer エンコーダー内のMulti-Head-Attention構造に続く2層MLP
    Attributes:
        linear1: 隠れ層
        linear2: 出力層
        activation: 活性化関数
    """
    def __init__(self, 
                 num_input_layer_units: int,
                 num_mlp_units: int):
        """2層の全結合層を定義
        Args:
            num_inputlayer(int): 特徴マップ生成時の全結合層のユニット数
            num_mlp_units(int): 多層パーセプトロンのユニット数
        """
        super().__init__()
        # 隠れ層
        self.linear1 = nn.Linear(num_input_layer_units, num_mlp_units)
        # 出力層
        self.linear2 = nn.Linear(num_mlp_units, num_input_layer_units)
        # 活性化関数はGELU
        self.activation = nn.GELU()
    
    def forward(self, x:torch.Tensor):
        """順伝播処理を行う
        Args:
            x(torch.Tensor): 特徴マップ(バッチサイズ, 特徴量数, 特徴量次元)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    """Encoderブロックの定義
    Attributes:
        attention: Multi-Head Self-Attention
        mlp: 2層構造の多層パーセプトロン
        norm1: 先頭に配置する正規化層
        norm2: Multi-Head-Self-Attentionの直後に配置する正規化層
    """
    def __init__(self, num_inputlayer_units: int,
                 num_heads: int,
                 num_mlp_units: int):
        """
        Args:
            num_input_units(int): 特徴マップ生成時の全結合層のユニット数
            num_heads(int): マルチヘッドアテンションのヘッド数
            num_mlp_units(int): 多層パーセプトロンのユニット数
        """
        super().__init__()
        # Multi-Head-Self-Attentionを生成
        self.attention = MultiHeadSelfAttention(num_inputlayer_units, num_heads)
        # MLPを作成
        self.mlp = MLP(num_inputlayer_units, num_mlp_units)
        # LayerNormを作成
        self.norm1 = nn.LayerNorm(num_inputlayer_units)
        self.norm2 = nn.LayerNorm(num_inputlayer_units)
    
    def forward(self, x: torch.Tensor):
        """順伝播を行う
        Args:
            x(torch.Tensor): 特徴マップ(バッチサイズ, 特徴量数, 特徴量次元)
        """
        x = self.norm1(x) # 正規化層
        x = self.attention(x) + x # 残差接続を実現
        x = self.norm2(x) # 正規化層
        x = self.mlp(x) + x # 残差接続

        return x



