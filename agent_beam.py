"""ビームサーチ (各局面で評価値の上位の手だけを深く読む) で着手を選ぶエージェント.

agent_ab.py / agent_abDeck.py のアルファベータ法を基にしている. 違いは, 探索の各局面で全ての合法手を
1手先の評価値で並べ, 上位 beam_width 個の手だけを次の深さで読む (それ以外の手はその場で打ち切る) 点である.
自分の手番では自分にとって評価値の高い手を, 相手の手番では相手にとって評価値の高い手を残す.
残した手の中では, 従来どおりアルファベータ法の枝刈りも行う.

- 探索開始時に, 相手の手札と山札の順番をランダムに決めた局面を num_samples 通り生成し,
  それぞれで探索して最善手に投票する (agent_abDeck.py と同じ)
- 最後の深さでは, 並べ替えのために計算した評価値をそのまま使うので, 全ての合法手を考慮する
- 山札が無くなった後は局面が確定するため, ビームサーチは使わず agent_ab.py と同じく終局まで読み切る
- 評価関数は agent_ab.py と同じものを使う

計算量の目安 (評価関数の呼び出し回数): num_samples × 合法手の数 × beam_width^(depth - 1)
(アルファベータ法では num_samples × 合法手の数^(depth / 2) 程度なので, 深い探索がしやすい)

実行時の指定 (match.py / play_vs_agent.py):
    python3 match.py 100 beam@depth=4,num_samples=4,beam_width=5 abdeck@depth=3,num_samples=4

使い方:
    import pybl
    from agent_beam import beam_move

    state = pybl.GameState()
    move = beam_move(state.observe(state.side_to_move), depth=4, num_samples=4, beam_width=5)
    state.play(move)
"""
import math
import random
from collections import defaultdict

import pybl

from agent_ab import _search_root, _terminal_score, evaluate

DEPTH = 4               # 山札が残っている間の探索の深さ (手数)
NUM_SAMPLES = 4         # 探索開始時に生成する局面 (相手の手札と山札の組み合わせ) の数
BEAM_WIDTH = 4          # 各局面で次の深さに進める手の数 (評価値の上位から)
ENDGAME_DEPTH = 20      # 山札が無くなった後の探索の深さ (終局まで読み切るのに十分な深さ)

_default_rng = random.Random()


def _scored_moves(state, ply: int) -> list:
    """全ての合法手を, 指した直後の局面の評価値 (手を指すプレイヤーから見た値) の高い順に並べて返す.

    返り値は (評価値, 着手) のリスト. ply は state の根からの手数 (終局時の評価値の補正に使う)
    """
    scored = []
    for move in state.get_legal_moves().tolist():
        state.play(move)
        value = -_terminal_score(state, ply + 1) if state.is_terminal else -evaluate(state)
        state.undo()
        scored.append((value, move))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _beam_negamax(state, depth: int, alpha: float, beta: float, ply: int, beam_width: int) -> float:
    if state.is_terminal:
        return _terminal_score(state, ply)
    if depth == 0:
        return evaluate(state)

    scored = _scored_moves(state, ply)
    if depth == 1:
        return scored[0][0]     # 最後の深さでは, 並べ替えに使った評価値がそのまま子の値になる

    best = -math.inf
    for _, move in scored[:beam_width]:
        state.play(move)
        value = -_beam_negamax(state, depth - 1, -beta, -alpha, ply + 1, beam_width)
        state.undo()
        if value > best:
            best = value
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break
    return best


def _beam_search_root(state, depth: int, beam_width: int):
    """state の手番のプレイヤーの最善手と評価値を, ビームサーチで求める"""
    scored = _scored_moves(state, 0)
    if depth == 1:
        value, move = scored[0]
        return move, value

    best_move, alpha = None, -math.inf
    for _, move in scored[:beam_width]:
        state.play(move)
        value = -_beam_negamax(state, depth - 1, -math.inf, -alpha, 1, beam_width)
        state.undo()
        if best_move is None or value > alpha:
            best_move, alpha = move, value
    return best_move, alpha


def beam_move(obs: pybl.Observation, depth: int | None = None, num_samples: int | None = None,
              beam_width: int | None = None, endgame_depth: int | None = None, rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, 着手をビームサーチで選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.
    depth: 山札が残っている間の探索の深さ (1以上. 省略時は DEPTH)
    num_samples: 探索開始時に生成する局面の数 (1以上. 省略時は NUM_SAMPLES)
    beam_width: 各局面で次の深さに進める手の数 (1以上. 省略時は BEAM_WIDTH)
    endgame_depth: 山札が無くなった後の探索の深さ (省略時は ENDGAME_DEPTH)
    rng: random.Random を渡すと, 局面の生成にその乱数を使う (再現性が必要な場合に用いる)

    置ける場所が無い場合は pybl.PASS_MOVE を返す. 終局している場合は ValueError を送出する.
    """
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    depth = DEPTH if depth is None else int(depth)
    num_samples = NUM_SAMPLES if num_samples is None else int(num_samples)
    beam_width = BEAM_WIDTH if beam_width is None else int(beam_width)
    endgame_depth = ENDGAME_DEPTH if endgame_depth is None else int(endgame_depth)
    if min(depth, num_samples, beam_width, endgame_depth) < 1:
        raise ValueError("depth, num_samples, beam_width and endgame_depth must be at least 1")
    rng = rng or _default_rng

    moves = obs.get_legal_moves()
    if len(moves) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")
    if len(moves) == 1:
        return moves[0]

    if obs.deck_count == 0:
        # 見えないカードは全て相手の手札なので局面が確定する. ビームサーチは使わず終局まで読み切る
        move, _ = _search_root(obs.sample_state(seed=rng.getrandbits(64)), endgame_depth)
        return pybl.Move(move)

    # 探索開始時に, 相手の手札と山札の順番をランダムに決めた局面を num_samples 通り生成する
    samples = [obs.sample_state(seed=rng.getrandbits(64)) for _ in range(num_samples)]

    votes = defaultdict(int)
    values = defaultdict(float)
    for sample in samples:
        move, value = _beam_search_root(sample, depth, beam_width)
        votes[move] += 1
        values[move] += value

    # 最も多くの局面で最善手となった手を選ぶ. 同数なら評価値の合計が大きい手を選ぶ
    best = max(votes, key=lambda m: (votes[m], values[m]))
    return pybl.Move(best)
