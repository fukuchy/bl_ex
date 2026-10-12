"""ルートのみ ISMCTS で着手を選ぶエージェント.

v1 の ISMCTS (agent_ismcts.py, doc/ismcts_spec.md) から, ルートより下の探索木だけを取り除いたもの.
v1 との違いがルートより下の木の有無だけになるよう, それ以外 (determinization, 同等手のまとめ, 選択式,
ロールアウト, 報酬, 着手決定, 反復回数と探索係数の既定値) は v1 と同じにしている.

| エージェント             | 手ごとのプレイアウト配分 | ルートより下の木 |
| モンテカルロ法 (agent_mon) | 均等                   | なし            |
| ルートのみ ISMCTS (本ファイル) | UCB1               | なし            |
| v1 ISMCTS (agent_ismcts)  | UCB1                   | あり            |

使い方:
    import pybl
    from agent_root import root_ismcts_move

    state = pybl.GameState()
    move = root_ismcts_move(state.observe(state.side_to_move))
    state.play(move)
"""
import math
import random

import pybl

from agent_ismcts import EXPLORATION, NUM_ITERATIONS, _reward, representative_moves

_default_rng = random.Random()


class MoveStats:
    """ルートの手 (同等手の代表) ごとの統計 (仕様 3.1)"""
    __slots__ = ("move", "visits", "reward", "availability")

    def __init__(self, move):
        self.move = move
        self.visits = 0         # 訪問回数 N
        self.reward = 0.0       # 報酬の合計 W
        self.availability = 1   # 選択可能回数 A (初めて試したときに1で始める)

    def ucb(self, c: float) -> float:
        return self.reward / self.visits + c * math.sqrt(math.log(self.availability) / self.visits)


def _search(obs: pybl.Observation, root_moves: list, num_iterations: int, c: float, rng: random.Random) -> dict:
    """ルートのみ ISMCTS で探索し, 着手 -> MoveStats を返す (仕様 3.2)"""
    me = obs.player
    stats = {}
    base = obs.sample_state(seed=rng.getrandbits(64))   # 基準の局面
    work = pybl.GameState(seed=0)                       # 作業用の局面

    for _ in range(num_iterations):
        # 1. determinization: 基準の局面を複製し, 自分から見えないカードを配り直す
        base.copy_to(work)
        work.determinize(me, seed=rng.getrandbits(64))

        # 2. 手の選択: 未試行の手があれば一様ランダム, 無ければ全候補の A を増やしてから UCB1 最大の手
        untried = [m for m in root_moves if m not in stats]
        if untried:
            move = untried[rng.randrange(len(untried))]
            stats[move] = MoveStats(move)
        else:
            for s in stats.values():
                s.availability += 1
            move = max(root_moves, key=lambda m: stats[m].ucb(c))

        # 3. 適用 / 4. ロールアウト (相手の手番以降はすべてロールアウトで進める)
        work.play(move)
        result = work.result if work.is_terminal else work.random_playout(seed=rng.getrandbits(64))

        # 5. 更新
        s = stats[move]
        s.visits += 1
        s.reward += _reward(result, me)

    return stats


def root_ismcts_move(obs: pybl.Observation, num_iterations: int = NUM_ITERATIONS, c: float = EXPLORATION,
                     rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, 着手をルートのみ ISMCTS で選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.
    num_iterations: 1手あたりの反復回数 (既定値は v1 の agent_ismcts.NUM_ITERATIONS と同じ)
    c: UCB1 の探索係数 (既定値は v1 の agent_ismcts.EXPLORATION と同じ)
    rng: random.Random を渡すと, 探索の全ての乱数をそこから作る (同じシードなら同じ手を返す)

    置ける場所が無い場合は pybl.PASS_MOVE を返す. 終局している場合は ValueError を送出する.
    """
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    rng = rng or _default_rng
    if len(obs.get_legal_moves()) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")

    root_moves = representative_moves(obs)
    if len(root_moves) == 1:
        return pybl.Move(root_moves[0])

    stats = _search(obs, root_moves, num_iterations, c, rng)

    # 訪問回数が最大の手. 同数なら平均報酬が大きい手
    best = max(stats.values(), key=lambda s: (s.visits, s.reward / s.visits))
    return pybl.Move(best.move)
