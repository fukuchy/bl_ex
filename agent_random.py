"""手番のプレイヤーの合法手からランダムに着手を選ぶエージェント.

使い方:
    import pybl
    from agent_random import random_move

    state = pybl.GameState()
    move = random_move(state.observe(state.side_to_move))
    state.play(move)
"""
import random

import pybl

_default_rng = random.Random()


def random_move(obs: pybl.Observation, rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, 着手を合法手からランダムに選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.
    置ける場所が無い場合は pybl.PASS_MOVE を返す. 終局している場合は ValueError を送出する.
    rng に random.Random を渡すと, その乱数で選ぶ (再現性が必要な場合に用いる).
    """
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    moves = obs.get_legal_moves()
    if len(moves) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")
    return moves[(rng or _default_rng).randrange(len(moves))]
