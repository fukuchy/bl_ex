"""学習したネットワーク (az/model.py) で着手を選ぶエージェント.

az_move (match.py では az): ネットワークを使う ISMCTS (az/mcts.py, AlphaZero 型) で探索して手を選ぶ
az_policy_move (azp): 探索を行わず, ネットワークの方策だけで手を選ぶ

実行時の指定 (match.py / play_vs_agent.py):
    python3 match.py 400 az abdeck -j 8
    python3 match.py 400 az@checkpoint=checkpoints/pretrain.pt,num_simulations=400,c_puct=2.0 abdeck -j 8
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
NUM_SIMULATIONS = 200                       # az_move の1手あたりのシミュレーションの回数
C_PUCT = 1.5                                # az_move の PUCT の探索係数
BATCH_SIZE = 16                             # az_move でまとめてネットワークで評価する葉の数

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


def az_move(obs: pybl.Observation, checkpoint: str = CHECKPOINT, num_simulations: int = NUM_SIMULATIONS,
            c_puct: float = C_PUCT, batch_size: int = BATCH_SIZE, fpu: float = 0.0, temperature: float = 0.0,
            rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, ネットワークを使う ISMCTS で着手を選んで返す.

    checkpoint: 学習したネットワークのファイル
    num_simulations: 1手あたりのシミュレーションの回数
    c_puct: PUCT の探索係数
    batch_size: まとめてネットワークで評価する葉の数
    fpu: 未訪問の手の Q の値
    temperature: 0 なら訪問回数が最大の手, 正なら訪問回数^(1/temperature) に比例した確率で手を選ぶ
    rng: random.Random を渡すと, 探索と手の選択の全ての乱数をそこから作る (同じシードなら同じ手を返す)

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

    from az.mcts import NetworkEvaluator, search
    rng = rng or _default_rng
    evaluator = _evaluators.get(checkpoint)
    if evaluator is None:
        evaluator = _evaluators[checkpoint] = NetworkEvaluator(_load(checkpoint), "cpu")
    result = search(obs, evaluator, num_simulations=int(num_simulations), c_puct=float(c_puct),
                    batch_size=int(batch_size), fpu=float(fpu), rng=rng)
    if len(result.moves) == 1:
        return pybl.Move(result.moves[0])
    return pybl.Move(result.sample_move(float(temperature), rng))


_evaluators = {}
