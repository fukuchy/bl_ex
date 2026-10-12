"""単純なモンテカルロ法で着手を選ぶエージェント.

各合法手について, 手番のプレイヤーから見える情報 (Observation) と矛盾しない局面を生成し (相手の手札と
山札は無作為に配分), その手を指してから終局までランダムに着手する (プレイアウト). これを繰り返し,
勝率 (引き分けは0.5勝) が最も高い手を選ぶ.

使い方:
    import pybl
    from agent_mon import montecarlo_move

    state = pybl.GameState()
    move = montecarlo_move(state.observe(state.side_to_move))
    state.play(move)
"""
import random

import pybl

_default_rng = random.Random()

NUM_PLAYOUT = 3000


def montecarlo_move(obs: pybl.Observation, num_playouts: int = NUM_PLAYOUT, rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, 着手をモンテカルロ法で選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.
    num_playouts: プレイアウトの総数. 各合法手に均等に割り振る (多いほど正確になるが遅くなる)
    rng: random.Random を渡すと, 局面の生成とプレイアウトにその乱数を使う (再現性が必要な場合に用いる)

    置ける場所が無い場合は pybl.PASS_MOVE を返す. 終局している場合は ValueError を送出する.
    """
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    rng = rng or _default_rng
    moves = obs.get_legal_moves()
    if len(moves) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")
    if len(moves) == 1:
        return moves[0]

    win = pybl.FIRST_WIN if obs.player == pybl.FIRST else pybl.SECOND_WIN
    scores = [0.0] * len(moves)
    playouts_per_move = max(1, num_playouts // len(moves))

    for k, move in enumerate(moves):
        for _ in range(playouts_per_move):
            sim = obs.sample_state(seed=rng.getrandbits(64))
            sim.play(move)
            result = sim.result if sim.is_terminal else sim.random_playout(seed=rng.getrandbits(64))
            if result == win:
                scores[k] += 1.0
            elif result == pybl.DRAW:
                scores[k] += 0.5

    # 各手のプレイアウト数は等しいため, 勝ち数 (引き分けは0.5) が最大の手が勝率最大の手になる
    return moves[max(range(len(moves)), key=lambda k: scores[k])]
