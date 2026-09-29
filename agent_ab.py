"""アルファベータ法 (negamax) で着手を選ぶエージェント.

相手の手札と山札の中身は見えないため, 手番のプレイヤーから見える情報 (Observation) と矛盾しない局面を
複数生成し (決定化), それぞれの局面でアルファベータ法による探索を行って最善手に投票する.
山札が無くなった後は見えないカードが全て相手の手札になり局面が確定するため, 終局まで読み切る.
(山札が無くなった時点で盤面の空きは高々8枚分であり, 実用的な時間で読み切れる)

使い方:
    import pybl
    from agent_ab import alphabeta_move

    state = pybl.GameState()
    move = alphabeta_move(state.observe(state.side_to_move))
    state.play(move)
"""
import math
import random
from collections import defaultdict

import pybl

WIN_SCORE = 100000          # 勝ちの評価値 (早く勝つほど大きくなるよう手数を引く)
CLAIMED_FLAG_SCORE = 100    # 確保済みフラッグ1本あたりの評価値
FLAG_DIFF_LIMIT = 64        # 未確保フラッグ1本あたりの評価値の上限 (フォーメーション2段階分)

_default_rng = random.Random()


def _side_potential(state, flag: int, player, available: int) -> int:
    """player がフラッグで作りうる最強のフォーメーションの強さ. 完成できなければ NULL_STRENGTH (-1)"""
    cards = state.get_flag_cards(flag, player)
    if len(cards) == pybl.FORMATION_SIZE:
        return state.get_flag_strength(flag, player)
    return pybl.best_completion(cards, available)


def evaluate(state) -> float:
    """手番のプレイヤーから見た局面の評価値 (終局していない局面用).

    確保済みフラッグの本数の差と, 未確保の各フラッグで両者が作りうる最強のフォーメーションの強さの差を合計する.
    """
    me = state.side_to_move
    opp = pybl.to_opponent(me)
    available = int(pybl.ALL_CARDS) & ~int(state.board_cards)

    score = CLAIMED_FLAG_SCORE * (state.get_claimed_flags(me).bit_count() - state.get_claimed_flags(opp).bit_count())
    for i in range(pybl.NUM_FLAGS):
        if state.get_flag_owner(i) != pybl.NULL_PLAYER:
            continue
        diff = _side_potential(state, i, me, available) - _side_potential(state, i, opp, available)
        score += max(-FLAG_DIFF_LIMIT, min(FLAG_DIFF_LIMIT, diff))
    return score


def _terminal_score(state, ply: int) -> float:
    """終局した局面の, 手番のプレイヤーから見た評価値"""
    winner = state.winner
    if winner == pybl.NULL_PLAYER:
        return 0
    return WIN_SCORE - ply if winner == state.side_to_move else -(WIN_SCORE - ply)


def _ordered_moves(state, moves) -> list:
    """1手先の評価値が高い順に並べる (枝刈りを効きやすくするため)"""
    def score(move):
        state.play(move)
        value = -_terminal_score(state, 1) if state.is_terminal else -evaluate(state)
        state.undo()
        return value
    return sorted(moves.tolist(), key=score, reverse=True)


def _negamax(state, depth: int, alpha: float, beta: float, ply: int) -> float:
    if state.is_terminal:
        return _terminal_score(state, ply)
    if depth == 0:
        return evaluate(state)

    moves = state.get_legal_moves()
    # 残り深さが2以上の局面でのみ手を並べ替える (葉の直前で並べ替えると評価の回数が増えるだけのため)
    moves = _ordered_moves(state, moves) if depth >= 2 else moves.tolist()

    best = -math.inf
    for move in moves:
        state.play(move)
        value = -_negamax(state, depth - 1, -beta, -alpha, ply + 1)
        state.undo()
        if value > best:
            best = value
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break
    return best


def _search_root(state, depth: int):
    """state の手番のプレイヤーの最善手と評価値を返す"""
    best_move, alpha = None, -math.inf
    for move in _ordered_moves(state, state.get_legal_moves()):
        state.play(move)
        value = -_negamax(state, depth - 1, -math.inf, -alpha, 1)
        state.undo()
        if best_move is None or value > alpha:
            best_move, alpha = move, value
    return best_move, alpha


def alphabeta_move(obs: pybl.Observation, depth: int = 2, num_samples: int = 8, endgame_depth: int = 20,
                   rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, 着手をアルファベータ法による探索で選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.

    depth: 山札が残っている間の探索の深さ (手数)
    num_samples: 山札が残っている間に生成する局面の数 (多いほど相手の手札の不確かさを考慮できるが遅くなる)
    endgame_depth: 山札が無くなった後の探索の深さ. 既定値は終局まで読み切るのに十分な深さ
    rng: random.Random を渡すと, 局面の生成にその乱数を使う (再現性が必要な場合に用いる)

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

    if obs.deck_count == 0:
        # 見えないカードは全て相手の手札なので, 生成される局面は実際の局面と一致する
        move, _ = _search_root(obs.sample_state(seed=rng.getrandbits(64)), endgame_depth)
        return pybl.Move(move)

    votes = defaultdict(int)
    values = defaultdict(float)
    for _ in range(num_samples):
        sample = obs.sample_state(seed=rng.getrandbits(64))
        move, value = _search_root(sample, depth)
        votes[move] += 1
        values[move] += value

    # 最も多くの局面で最善手となった手を選ぶ. 同数なら評価値の合計が大きい手を選ぶ
    best = max(votes, key=lambda m: (votes[m], values[m]))
    return pybl.Move(best)
