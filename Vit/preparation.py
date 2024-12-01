from tqdm import tqdm
import torch

from Mymodel import VisionTransformer

class ModelConfig:
    """パラメータの初期値を設定するクラス
    """
    def __init__(self):
        self.num_epochs = 150 # 学習回数(エポック数)
        self.batch_size = 32 # バッチサイズ
        self.lr = 0.01 # 学習率
        self.img_size = 32 # 画像サイズ
        self.patch_size = 4 # パッチサイズ
        self.num_inputlayers = 512 # 入力層のユニット数
        self.num_heads = 8 # ヘッド数
        self.num_mlp_units = 512 # MLPのユニット数
        self.num_layers = 6 # Encoderの層数


def evalate(data_loader, model, loss_func):
    """検証を行う
    Args:
        data_loader(DataLoader): テスト用のデータローダー
        model(nn.Module): Vision Transformerモデル
        loss_func(nn.functional): 損失関数
    """
    # モデルを検証モード切り替え
    model.eval()

    losses = [] # バッチごとの損失を格納するリスト
    correct_preds = 0 # バッチごとの正解数をカウントする変数
    total_samples = 0 # バッチごとのサンプル数をカウントする変数

    # dataローダーからバッチを取り出して学習中のモデルで検証を行う
    for x, y in tqdm(data_loader, desc='Evaluating', total=len(data_loader), leave=False):
        with torch.no_grad():
            # デバイスへの転送
            x, y = x.to(device=model.get_device()), y.to(device=model.get_device())
            # モデルへの入力
            preds = model(x) 
            # preds: [batch size(32), num_classes(10)]

            # 損失の計算
            loss = loss_func(preds, y)
            # 損失をリストに格納
            losses.append(loss.item())
            # 予測結果の取得
            predicted = torch.max(preds, dim=1)[1] # predicted: [batch size(32)]
            # 正解数をカウント
            correct_preds += (predicted == y).sum().item()
            # サンプル数をカウント
            total_samples += y.size(0)

    # 平均損失の計算
    avg_loss = sum(losses) / len(losses)
    # 正解率の計算
    accuracy = correct_preds / total_samples

    return avg_loss, accuracy





