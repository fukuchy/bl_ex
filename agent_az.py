"""学習したネットワーク (az/model.py) で着手を選ぶエージェント.

現在は探索を行わず, ネットワークの方策だけで手を選ぶ (az_policy_move). ネットワークを使う ISMCTS
(AlphaZero 型) は今後追加する.

実行時の指定 (match.py / play_vs_agent.py):
    python3 match.py 400 azp random -j 8
    python3 match.py 400 azp@checkpoint=checkpoints/pretrain.pt,temperature=0.5 abdeck -j 8

使い方:
    import pybl
    from agent_az import az_policy_move

    state = pybl.GameState()
    move = az_policy_move(state.observe(state.side_to_move), checkpoint="checkpoints/pretrain.pt")
    state.play(move)
"""
import random
from pathlib import Path

import numpy as np

import pybl
from az import features as F

CHECKPOINT = "checkpoints/pretrain.pt"      # 既定のチェックポイント (bl_ex のディレクトリからの相対パスでもよい)

_default_rng = random.Random()
_models = {}


def _load(checkpoint: str):
    """チェックポイントを読み込む (プロセスごとに1回だけ). 対戦の並列実行を妨げないよう CPU で1スレッドで動かす"""
    path = Path(checkpoint)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent / path
    key = str(path)
    if key not in _models:
        import torch
        from az.model import load_checkpoint
        torch.set_num_threads(1)
        _models[key] = load_checkpoint(path, device="cpu")[0]
    return _models[key]


def az_policy_move(obs: pybl.Observation, checkpoint: str = CHECKPOINT, temperature: float = 0.0,
                   rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, ネットワークの方策だけで着手を選んで返す.

    checkpoint: 学習したネットワークのファイル
    temperature: 0 なら方策の確率が最大の手, 正なら確率^(1/temperature) に比例した確率で手を選ぶ
    rng: random.Random を渡すと, temperature > 0 のときの手の選択にその乱数を使う

    置ける場所が無い場合は pybl.PASS_MOVE を返す. 終局している場合は ValueError を送出する.
    """
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    moves = obs.get_legal_moves()
    if len(moves) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")
    if len(moves) == 1:
        return moves[0]

    from az.model import predict
    model = _load(checkpoint)
    planes, scalars = F.encode(obs)
    probs, _ = predict(model, planes[None], scalars[None], F.legal_mask(obs)[None])
    probs = probs[0][moves]
    if temperature <= 0:
        return moves[int(np.argmax(probs))]
    weights = probs ** (1.0 / temperature)
    return moves[(rng or _default_rng).choices(range(len(moves)), weights=weights)[0]]
