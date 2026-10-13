"""探索の深さを実行時に指定できるアルファベータ法のエージェント (agent_ab.py の改良版).

探索開始時に, 手番のプレイヤーから見える情報 (Observation) と矛盾しない局面 (相手の手札と山札の順番を
ランダムに決めたもの) を num_samples 通り生成し, それぞれでアルファベータ法による探索を行って最善手に投票する.
評価関数・探索の処理は agent_ab.py と同じものを使う.

agent_ab.py との違い:
- 探索の深さ (depth) と生成する局面の数 (num_samples) を実行時に指定できる.
  match.py / play_vs_agent.py では, エージェント名の後ろに @引数名=値 を付けて指定する.
      python3 match.py 100 abdeck@depth=3 ab
      python3 play_vs_agent.py abdeck@depth=3,num_samples=16
- 指定しない場合は, このファイルの DEPTH, NUM_SAMPLES の値を使う

使い方:
    import pybl
    from agent_abDeck import abdeck_move

    state = pybl.GameState()
    move = abdeck_move(state.observe(state.side_to_move), depth=3)
    state.play(move)
"""
import random
from collections import defaultdict

import pybl

from agent_ab import _search_root

DEPTH = 2               # 山札が残っている間の探索の深さ (手数. 自分の手と相手の応手で2)
NUM_SAMPLES = 8         # 探索開始時に生成する局面 (相手の手札と山札の組み合わせ) の数
ENDGAME_DEPTH = 20      # 山札が無くなった後の探索の深さ (終局まで読み切るのに十分な深さ)

_default_rng = random.Random()


def abdeck_move(obs: pybl.Observation, depth: int | None = None, num_samples: int | None = None,
                endgame_depth: int | None = None, rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, 着手をアルファベータ法による探索で選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.
    depth: 山札が残っている間の探索の深さ (1以上. 省略時は DEPTH)
    num_samples: 探索開始時に生成する局面の数 (1以上. 省略時は NUM_SAMPLES)
    endgame_depth: 山札が無くなった後の探索の深さ (省略時は ENDGAME_DEPTH)
    rng: random.Random を渡すと, 局面の生成にその乱数を使う (再現性が必要な場合に用いる)

    置ける場所が無い場合は pybl.PASS_MOVE を返す. 終局している場合は ValueError を送出する.
    """
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    depth = DEPTH if depth is None else int(depth)
    num_samples = NUM_SAMPLES if num_samples is None else int(num_samples)
    endgame_depth = ENDGAME_DEPTH if endgame_depth is None else int(endgame_depth)
    if depth < 1 or num_samples < 1 or endgame_depth < 1:
        raise ValueError("depth, num_samples and endgame_depth must be at least 1")
    rng = rng or _default_rng

    moves = obs.get_legal_moves()
    if len(moves) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")
    if len(moves) == 1:
        return moves[0]

    if obs.deck_count == 0:
        # 見えないカードは全て相手の手札なので, 生成される局面は実際の局面と一致する
        move, _ = _search_root(obs.sample_state(seed=rng.getrandbits(64)), endgame_depth)
        return pybl.Move(move)

    # 探索開始時に, 相手の手札と山札の順番をランダムに決めた局面を num_samples 通り生成する
    samples = [obs.sample_state(seed=rng.getrandbits(64)) for _ in range(num_samples)]

    votes = defaultdict(int)
    values = defaultdict(float)
    for sample in samples:
        move, value = _search_root(sample, depth)
        votes[move] += 1
        values[move] += value

    # 最も多くの局面で最善手となった手を選ぶ. 同数なら評価値の合計が大きい手を選ぶ
    best = max(votes, key=lambda m: (votes[m], values[m]))
    return pybl.Move(best)
