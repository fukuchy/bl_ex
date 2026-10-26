"""学習したネットワーク (az/model.py) で着手を選ぶエージェント.

az_move (match.py では az): ネットワークを使う ISMCTS (az/mcts.py, AlphaZero 型) で探索して手を選ぶ
az_policy_move (azp): 探索を行わず, ネットワークの方策だけで手を選ぶ
az_oracle_move (azo): 診断用. az_move と同じ探索で, 決定化の代わりに本当の相手の手札 (reveal=all なら山札の順番も) を使う

実行時の指定 (match.py / play_vs_agent.py):
    python3 match.py 400 az abdeck -j 8
    python3 match.py 400 az@checkpoint=checkpoints/pretrain.pt,num_simulations=400,c_puct=2.0 abdeck -j 8
    python3 match.py 400 az@fpu_reduction=0.2,c_puct=1.0 az -j 8
    python3 match.py 400 az@device=cuda abdeck -j 8         # 推論を GPU で行う (auto なら GPU があれば cuda)
    python3 match.py 400 azp random -j 8
    python3 match.py 400 azo@checkpoint=checkpoints/pretrain.pt,device=cuda az@checkpoint=checkpoints/pretrain.pt,device=cuda -j 8
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


def _device(device: str) -> str:
    """推論に使う装置. auto なら GPU (cuda) があれば cuda, 無ければ cpu"""
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load(checkpoint: str, device: str = "cpu"):
    """チェックポイントを読み込む (プロセスと装置ごとに1回だけ). 対戦の並列実行を妨げないよう CPU では1スレッドで動かす"""
    path = Path(checkpoint)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).resolve().parent / path
    key = (str(path), device)
    if key not in _models:
        import torch
        from az.model import load_checkpoint
        torch.set_num_threads(1)
        _models[key] = load_checkpoint(path, device=device)[0]
    return _models[key]


def az_policy_move(obs: pybl.Observation, checkpoint: str = CHECKPOINT, temperature: float = 0.0,
                   device: str = "cpu", rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, ネットワークの方策だけで着手を選んで返す.

    checkpoint: 学習したネットワークのファイル
    temperature: 0 なら方策の確率が最大の手, 正なら確率^(1/temperature) に比例した確率で手を選ぶ
    device: 推論に使う装置 (cpu / cuda / auto. auto なら GPU があれば cuda)
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
    device = _device(device)
    model = _load(checkpoint, device)
    planes, scalars = F.encode(obs)
    probs, _ = predict(model, planes[None], scalars[None], F.legal_mask(obs)[None], device=device)
    probs = probs[0][moves]
    if temperature <= 0:
        return moves[int(np.argmax(probs))]
    weights = probs ** (1.0 / temperature)
    return moves[(rng or _default_rng).choices(range(len(moves)), weights=weights)[0]]


def az_move(obs: pybl.Observation, checkpoint: str = CHECKPOINT, num_simulations: int = NUM_SIMULATIONS,
            c_puct: float = C_PUCT, batch_size: int = BATCH_SIZE, fpu: float = 0.0,
            fpu_reduction: float | None = None, temperature: float = 0.0, device: str = "cpu",
            rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, ネットワークを使う ISMCTS で着手を選んで返す.

    checkpoint: 学習したネットワークのファイル
    num_simulations: 1手あたりのシミュレーションの回数
    c_puct: PUCT の探索係数
    batch_size: まとめてネットワークで評価する葉の数
    fpu: 未訪問の手の Q の値
    fpu_reduction: 指定すると, 未訪問の手の Q を 訪問済みの手の平均価値 − fpu_reduction × √(訪問済みの手の事前確率の合計) にする
    temperature: 0 なら訪問回数が最大の手, 正なら訪問回数^(1/temperature) に比例した確率で手を選ぶ
    device: 推論に使う装置 (cpu / cuda / auto. auto なら GPU があれば cuda. cuda では半精度で推論する)
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
    device = _device(device)
    evaluator = _evaluators.get((checkpoint, device))
    if evaluator is None:
        evaluator = _evaluators[(checkpoint, device)] = NetworkEvaluator(_load(checkpoint, device), device)
    result = search(obs, evaluator, num_simulations=int(num_simulations), c_puct=float(c_puct),
                    batch_size=int(batch_size), fpu=float(fpu),
                    fpu_reduction=None if fpu_reduction is None else float(fpu_reduction), rng=rng)
    if len(result.moves) == 1:
        return pybl.Move(result.moves[0])
    return pybl.Move(result.sample_move(float(temperature), rng))


_evaluators = {}


def az_oracle_move(obs: pybl.Observation, state: pybl.GameState, reveal: str = "hand", checkpoint: str = CHECKPOINT,
                   num_simulations: int = NUM_SIMULATIONS, c_puct: float = C_PUCT, batch_size: int = BATCH_SIZE,
                   fpu: float = 0.0, fpu_reduction: float | None = None, temperature: float = 0.0,
                   device: str = "cpu", rng: random.Random | None = None):
    """診断用: az_move と同じ探索で, 見えないカードを配り直す代わりに本当の局面 state の情報を使って手を選ぶ.

    「相手の手札が分かっていたら, どれだけ強くなるか」(決定化を改善する余地の上限) を測るためのもので, ずるをする.
    reveal: hand なら相手の手札だけ本当のものを使い, 山札の順番はシミュレーションごとにランダムにする.
            all なら山札の順番も含めて本当の局面を使う
    state: 本当の局面 (match.py が写しを渡す). その他の引数は az_move と同じ
    """
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    if state.observe(obs.player) != obs:
        raise ValueError("state does not match obs")
    moves = obs.get_legal_moves()
    if len(moves) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")
    if len(moves) == 1:
        return moves[0]

    from az.mcts import NetworkEvaluator, revealed_world_sampler, search
    rng = rng or _default_rng
    device = _device(device)
    evaluator = _evaluators.get((checkpoint, device))
    if evaluator is None:
        evaluator = _evaluators[(checkpoint, device)] = NetworkEvaluator(_load(checkpoint, device), device)
    result = search(obs, evaluator, num_simulations=int(num_simulations), c_puct=float(c_puct),
                    batch_size=int(batch_size), fpu=float(fpu),
                    fpu_reduction=None if fpu_reduction is None else float(fpu_reduction),
                    sample_world=revealed_world_sampler(state, str(reveal)), rng=rng)
    if len(result.moves) == 1:
        return pybl.Move(result.moves[0])
    return pybl.Move(result.sample_move(float(temperature), rng))
