"""AlphaZero 型のネットワーク (方策と価値) と, その保存・読み込み・推論.

ネットワークは2種類ある (チェックポイントの config["arch"] で区別する).

TokenNet (arch="token", 既定): カードとフラッグをトークンとして扱う Transformer
    トークン: カード 60 個 (所在 20 種類 + 色 6 + 数字 10), フラッグ 9 個 (確保者・先に役を完成させたプレイヤー・
             位置・各陣営の枚数), 全体 1 個 (山札の枚数など). planes / scalars から組み立てる
    → Transformer エンコーダ (pre-LN) × layers
    → 方策: 「カード c をフラッグ f に置く」手のロジット = (カード c の射影) · (フラッグ f の射影) / √k,
            パスのロジットは全体のトークンから求める
    → 価値: 全体のトークンから求める (tanh)
    アテンションで「同じ色・連番」などのカード同士の関係を直接学べ, 方策の計算も全てのカード・フラッグで
    同じ重みを使うので, 全結合だけのネットワークより少ないデータで汎化しやすい.

BattleLineNet (arch="mlp") の構造 (MLP-ResNet):
    入力: planes (20x6x10) を平らにしたもの (1200) と scalars (41) をつなげた 1241 次元
    → 全結合で hidden 次元に変換 → 残差ブロック × blocks
    → 方策ヘッド: NUM_ACTIONS (541) 個のロジット (合法手以外はマスクしてから softmax する)
    → 価値ヘッド: 手番のプレイヤーから見た勝ちやすさ (-1〜1, tanh)

残差ブロック: x + Linear(ReLU(LayerNorm(Linear(ReLU(LayerNorm(x))))))  (pre-activation)
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as TF

from az import features as F

INPUT_DIM = F.NUM_PLANES * F.NUM_CARDS + F.NUM_SCALARS
DEFAULT_CONFIG = {"hidden": 512, "blocks": 6}
DEFAULT_TOKEN_CONFIG = {"d_model": 128, "layers": 4, "heads": 4}


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        h = self.fc1(TF.relu(self.norm1(x)))
        h = self.fc2(TF.relu(self.norm2(h)))
        return x + h


class BattleLineNet(nn.Module):
    def __init__(self, hidden: int = DEFAULT_CONFIG["hidden"], blocks: int = DEFAULT_CONFIG["blocks"]):
        super().__init__()
        self.config = {"arch": "mlp", "hidden": hidden, "blocks": blocks}
        self.input = nn.Linear(INPUT_DIM, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.policy_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, F.NUM_ACTIONS))
        self.value_head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, planes: torch.Tensor, scalars: torch.Tensor):
        """planes: (B, 20, 6, 10), scalars: (B, 41) → (方策のロジット (B, 541), 価値 (B,))"""
        x = torch.cat([planes.flatten(1), scalars], dim=1)
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        x = TF.relu(self.norm(x))
        return self.policy_head(x), torch.tanh(self.value_head(x)).squeeze(-1)


_CARD_FEATURES = F.NUM_PLANES + F.NUM_COLORS + F.NUM_RANKS        # 所在 20 + 色 6 + 数字 10
_FLAG_FEATURES = 4 + F.NUM_FLAGS + 2                                # 確保者・先に完成 4 + 位置 9 + 枚数 2
_GLOBAL_FEATURES = F.NUM_SCALARS - F.SCALAR_DECK                    # 山札の枚数など 5


class TokenNet(nn.Module):
    """カードとフラッグをトークンとして扱う Transformer のネットワーク (入出力は BattleLineNet と同じ)"""

    def __init__(self, d_model: int = DEFAULT_TOKEN_CONFIG["d_model"], layers: int = DEFAULT_TOKEN_CONFIG["layers"],
                 heads: int = DEFAULT_TOKEN_CONFIG["heads"]):
        super().__init__()
        self.config = {"arch": "token", "d_model": d_model, "layers": layers, "heads": heads}
        # カードの色と数字, フラッグの位置の one-hot (定数)
        cards = torch.arange(F.NUM_CARDS)
        self.register_buffer("card_identity", torch.cat([TF.one_hot(cards // F.NUM_RANKS, F.NUM_COLORS),
                                                         TF.one_hot(cards % F.NUM_RANKS, F.NUM_RANKS)], 1).float())
        self.register_buffer("flag_position", torch.eye(F.NUM_FLAGS))
        self.card_embed = nn.Linear(_CARD_FEATURES, d_model)
        self.flag_embed = nn.Linear(_FLAG_FEATURES, d_model)
        self.global_embed = nn.Linear(_GLOBAL_FEATURES, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.policy_card = nn.Linear(d_model, d_model)
        self.policy_flag = nn.Linear(d_model, d_model)
        self.policy_pass = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def tokens(self, planes: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        b = planes.shape[0]
        location = planes.reshape(b, F.NUM_PLANES, F.NUM_CARDS).transpose(1, 2)            # (B, 60, 20)
        card = torch.cat([location, self.card_identity.expand(b, -1, -1)], 2)
        flag_scalars = scalars[:, :F.SCALAR_DECK].reshape(b, 4, F.NUM_FLAGS).transpose(1, 2)  # (B, 9, 4)
        counts = torch.stack([location[:, :, F.PLANE_MY_FLAG:F.PLANE_OPP_FLAG].sum(1),
                              location[:, :, F.PLANE_OPP_FLAG:].sum(1)], 2) / 3.0           # (B, 9, 2)
        flag = torch.cat([flag_scalars, self.flag_position.expand(b, -1, -1), counts], 2)
        glob = scalars[:, F.SCALAR_DECK:].unsqueeze(1)                                        # (B, 1, 5)
        return torch.cat([self.card_embed(card), self.flag_embed(flag), self.global_embed(glob)], 1)

    def forward(self, planes: torch.Tensor, scalars: torch.Tensor):
        """planes: (B, 20, 6, 10), scalars: (B, 41) → (方策のロジット (B, 541), 価値 (B,))"""
        h = self.norm(self.encoder(self.tokens(planes, scalars)))
        card, flag, glob = h[:, :F.NUM_CARDS], h[:, F.NUM_CARDS:F.NUM_CARDS + F.NUM_FLAGS], h[:, -1]
        pair = torch.einsum("bcd,bfd->bcf", self.policy_card(card), self.policy_flag(flag))
        pair = pair / (card.shape[-1] ** 0.5)
        logits = torch.cat([pair.reshape(pair.shape[0], -1), self.policy_pass(glob)], 1)   # card * 9 + flag, パス
        return logits, torch.tanh(self.value_head(glob)).squeeze(-1)


def build_model(config: dict) -> nn.Module:
    """config (チェックポイントの "config") からネットワークを作る"""
    config = dict(config)
    arch = config.pop("arch", "mlp")
    if arch == "token":
        return TokenNet(**config)
    if arch == "mlp":
        return BattleLineNet(**config)
    raise ValueError(f"unknown arch: {arch}")


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """合法手以外を除いた log softmax (合法手以外は -inf ではなく十分小さい値にして NaN を避ける)"""
    return TF.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)


def loss_fn(logits, value, target_policy, target_value, mask, value_weight: float = 1.0):
    """AlphaZero の損失: value_weight × 価値の二乗誤差 + 方策の交差エントロピー. (合計, 価値の損失, 方策の損失) を返す"""
    value_loss = TF.mse_loss(value, target_value)
    policy_loss = -(target_policy * masked_log_softmax(logits, mask)).sum(dim=-1).mean()
    return value_weight * value_loss + policy_loss, value_loss, policy_loss


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path, model: nn.Module, **extra) -> None:
    """ネットワークの構造と重みを保存する (extra には学習の情報などを入れる)"""
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple:
    """保存したネットワークを読み込み, (ネットワーク, チェックポイントの内容) を返す (評価モード)"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict(model: nn.Module, planes: np.ndarray, scalars: np.ndarray, masks: np.ndarray,
            device: torch.device | str = "cpu") -> tuple:
    """numpy の入力 (バッチ) から, (合法手だけの方策の確率 (B, 541), 価値 (B,)) を numpy で返す"""
    p = torch.as_tensor(planes, dtype=torch.float32, device=device)
    s = torch.as_tensor(scalars, dtype=torch.float32, device=device)
    m = torch.as_tensor(masks, dtype=torch.bool, device=device)
    logits, value = model(p, s)
    probs = masked_log_softmax(logits, m).exp()
    return probs.cpu().numpy(), value.cpu().numpy()


class SymmetryAugmenter:
    """学習のバッチに, 要素ごとにランダムな対称変換を GPU 上でまとめて施す"""

    def __init__(self, device: torch.device | str):
        tables = F.symmetry_tables()
        self.tables = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in tables.items()}
        self.count = len(tables["action"])

    def __call__(self, planes, scalars, policy, mask, generator: torch.Generator | None = None):
        b = planes.shape[0]
        sym = torch.randint(self.count, (b,), device=planes.device, generator=generator)
        flat = planes.reshape(b, F.NUM_PLANES, F.NUM_CARDS)
        flat = flat.gather(1, self.tables["channel"][sym][:, :, None].expand(b, F.NUM_PLANES, F.NUM_CARDS))
        flat = flat.gather(2, self.tables["card"][sym][:, None, :].expand(b, F.NUM_PLANES, F.NUM_CARDS))
        action = self.tables["action"][sym]
        return (flat.reshape(planes.shape), scalars.gather(1, self.tables["scalar"][sym]),
                policy.gather(1, action), mask.gather(1, action))
