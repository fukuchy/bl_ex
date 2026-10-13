"""SO-ISMCTS (Single Observer Information Set MCTS, Cowling et al. 2012) で着手を選ぶエージェント.

仕様は doc/ismcts_spec.md (v1) に従う. 概要:
- 反復ごとに, 自分から見えないカード (相手の手札と山札) を一様ランダムに配り直した局面 (determinization) を用いる
- 選択は availability count 付き UCB1, 展開は1反復1ノード, ロールアウトは PyBL のランダムプレイアウト
- 報酬は勝ち 1・引き分け 0.5・負け 0. 指定回数の反復後, ルートで訪問回数が最大の手を返す
- ルートの自分の手に限り, フラッグの左右対称と色の入れ替えで厳密に同等な手を1つの代表にまとめる

使い方:
    import pybl
    from agent_ismcts import ismcts_move

    state = pybl.GameState()
    move = ismcts_move(state.observe(state.side_to_move))
    state.play(move)
"""
import math
import random

import pybl

NUM_ITERATIONS = 30000  # 1手あたりの反復回数 (仕様 7章. 初期値 1,000 から変更)
EXPLORATION = 0.1       # UCB1 の探索係数 c (仕様 7章)

_default_rng = random.Random()


class Node:
    """探索木のノード (仕様 5章). トランスポジションは扱わない"""
    __slots__ = ("move", "player", "parent", "children", "visits", "reward", "availability")

    def __init__(self, move, player, parent):
        self.move = move                # このノードに至った着手 (ルートは None)
        self.player = player            # 手を指したプレイヤー (報酬はこの視点. ルートは自分の相手)
        self.parent = parent
        self.children = {}              # 着手 -> 子ノード
        self.visits = 0                 # 訪問回数 N
        self.reward = 0.0               # 報酬の合計 W
        self.availability = 1           # 選択可能回数 A (展開時に1で始める)

    def ucb(self, c: float) -> float:
        return self.reward / self.visits + c * math.sqrt(math.log(self.availability) / self.visits)


def _reward(result, player) -> float:
    """終局結果の, player から見た報酬 (仕様 6.5)"""
    if result == pybl.DRAW:
        return 0.5
    winner = pybl.FIRST if result == pybl.FIRST_WIN else pybl.SECOND
    return 1.0 if winner == player else 0.0


# ---------------------------------------------------------------------------
# ルートの同等手のまとめ (仕様 6.6)
# ---------------------------------------------------------------------------

def _flag_signature(obs: pybl.Observation, flag: int) -> tuple:
    """左右対称の判定に用いるフラッグの状態 (両陣営の配置カード, 確保者, 先に役を完成させたプレイヤー)"""
    return (
        tuple(sorted(int(c) for c in obs.get_flag_cards(flag, pybl.FIRST))),
        tuple(sorted(int(c) for c in obs.get_flag_cards(flag, pybl.SECOND))),
        int(obs.get_flag_owner(flag)),
        int(obs.get_flag_first_completer(flag)),
    )


def _flag_representatives(obs: pybl.Observation) -> list:
    """各フラッグの代表のフラッグ番号. 盤面全体が左右対称なら i と 8-i を小さい方にまとめる"""
    last = pybl.NUM_FLAGS - 1
    symmetric = all(_flag_signature(obs, i) == _flag_signature(obs, last - i) for i in range(pybl.NUM_FLAGS // 2))
    if not symmetric:
        return list(range(pybl.NUM_FLAGS))
    return [min(i, last - i) for i in range(pybl.NUM_FLAGS)]


def _card_locations(obs: pybl.Observation) -> list:
    """各カードの所在 ("hand", "unseen", (フラッグ番号, 自分側か)) のリスト"""
    locations = [None] * pybl.NUM_CARDS
    for card in pybl.CardIterator(obs.hand):
        locations[int(card)] = "hand"
    for card in pybl.CardIterator(obs.unseen_cards):
        locations[int(card)] = "unseen"
    for flag in range(pybl.NUM_FLAGS):
        for player in (pybl.FIRST, pybl.SECOND):
            for card in obs.get_flag_cards(flag, player):
                locations[int(card)] = (flag, player == obs.player)
    return locations


def _color_representatives(obs: pybl.Observation) -> list:
    """各色の代表の色. 全ての数字でカードの所在が一致する色どうしを, 色番号の最小の色にまとめる"""
    locations = _card_locations(obs)
    pattern_to_color = {}
    representatives = []
    for color in range(pybl.NUM_COLORS):
        pattern = tuple(locations[int(pybl.make_card(color, rank))] for rank in range(1, pybl.NUM_RANKS + 1))
        representatives.append(pattern_to_color.setdefault(pattern, color))
    return representatives


def representative_moves(obs: pybl.Observation) -> list:
    """obs の合法手を, 同等な手を1つの代表にまとめて返す (代表の手は全て合法手)"""
    flag_rep = _flag_representatives(obs)
    color_rep = _color_representatives(obs)

    representatives = []
    seen = set()
    for move in obs.get_legal_moves().tolist():
        if move != pybl.PASS_MOVE:
            card = pybl.move_card(move)
            rep_card = pybl.make_card(color_rep[int(pybl.card_color(card))], pybl.card_rank(card))
            move = int(pybl.make_move(rep_card, flag_rep[pybl.move_flag(move)]))
        if move not in seen:
            seen.add(move)
            representatives.append(move)
    return representatives


# ---------------------------------------------------------------------------
# 探索 (仕様 6.1 - 6.5)
# ---------------------------------------------------------------------------

def _search(obs: pybl.Observation, root_moves: list, num_iterations: int, c: float, rng: random.Random) -> Node:
    """SO-ISMCTS で探索し, ルートノードを返す"""
    me = obs.player
    root = Node(None, pybl.to_opponent(me), None)
    base = obs.sample_state(seed=rng.getrandbits(64))   # 基準の局面
    work = pybl.GameState(seed=0)                       # 作業用の局面

    for _ in range(num_iterations):
        # 1. determinization: 基準の局面を複製し, 自分から見えないカードを配り直す
        base.copy_to(work)
        work.determinize(me, seed=rng.getrandbits(64))

        # 2. 選択 / 3. 展開
        node = root
        while not work.is_terminal:
            legal = root_moves if node is root else work.get_legal_moves().tolist()
            untried = [m for m in legal if m not in node.children]
            if untried:
                move = untried[rng.randrange(len(untried))]
                child = Node(move, work.side_to_move, node)
                node.children[move] = child
                work.play(move)
                node = child
                break

            candidates = [node.children[m] for m in legal]
            for child in candidates:
                child.availability += 1
            node = max(candidates, key=lambda n: n.ucb(c))
            work.play(node.move)

        # 4. ロールアウト
        result = work.result if work.is_terminal else work.random_playout(seed=rng.getrandbits(64))

        # 5. 逆伝播
        while node is not None:
            node.visits += 1
            node.reward += _reward(result, node.player)
            node = node.parent

    return root


def ismcts_move(obs: pybl.Observation, num_iterations: int = NUM_ITERATIONS, c: float = EXPLORATION,
                rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, 着手を SO-ISMCTS で選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.
    num_iterations: 1手あたりの反復回数
    c: UCB1 の探索係数 (報酬 0〜1 前提)
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

    root = _search(obs, root_moves, num_iterations, c, rng)

    # 訪問回数が最大の手. 同数なら平均報酬が大きい手 (仕様 6.7)
    best = max(root.children.values(), key=lambda n: (n.visits, n.reward / n.visits))
    return pybl.Move(best.move)
