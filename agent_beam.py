"""ビームサーチ (各深さで評価値の上位 beam_width 個のノードだけを残す) で着手を選ぶエージェント.

agent_ab.py / agent_abDeck.py のアルファベータ法のエージェントを基にしている (評価関数, 局面の生成,
最善手への投票, 山札が無くなった後の読み切りは同じ). 探索は次のように行う.

1. 展開: その深さに残したノード (根の深さでは根だけ) の全ての合法手を指し, 評価関数で採点する
2. 選択: 展開した全ての子の中から, その手を指したプレイヤーから見て評価値が高い上位 beam_width 個だけを
   次の深さで展開する (自分の手番の深さでは自分の有力な手, 相手の手番の深さでは相手の有力な応手を残す)
3. 値の計算: 作った木の上で, 葉から根に向かってミニマックス法で値を求める. 展開しなかった子は
   採点した評価値をそのまま使うので, 各ノードの値はそのノードの全ての合法手を考慮した値になる

各深さに残すノードは beam_width 個だけなので, 計算量 (評価関数の呼び出し回数) は深さに比例する:
    num_samples × 合法手の数 (最大63) × (1 + beam_width × (depth - 1))

- 探索開始時に, 相手の手札と山札の順番をランダムに決めた局面を num_samples 通り生成し,
  それぞれで探索して最善手に投票する (agent_abDeck.py と同じ)
- 山札が無くなった後は局面が確定するため, ビームサーチは使わず agent_ab.py と同じく終局まで読み切る

実行時の指定 (match.py / play_vs_agent.py):
    python3 match.py 100 beam@depth=6,num_samples=4,beam_width=8 abdeck@depth=3,num_samples=4

使い方:
    import pybl
    from agent_beam import beam_move

    state = pybl.GameState()
    move = beam_move(state.observe(state.side_to_move), depth=6, num_samples=4, beam_width=8)
    state.play(move)
"""
import random
from collections import defaultdict

import pybl

from agent_ab import _search_root, _terminal_score, evaluate

DEPTH = 6               # 山札が残っている間の探索の深さ (手数)
NUM_SAMPLES = 4         # 探索開始時に生成する局面 (相手の手札と山札の組み合わせ) の数
BEAM_WIDTH = 8          # 各深さで次の深さに進める (展開する) ノードの数
ENDGAME_DEPTH = 20      # 山札が無くなった後の探索の深さ (終局まで読み切るのに十分な深さ)

_default_rng = random.Random()


class _Node:
    """ビームサーチの木のノード"""
    __slots__ = ("move", "parent", "score", "terminal", "children", "state")

    def __init__(self, move, parent, score: float, terminal: bool):
        self.move = move            # このノードに至った着手 (根は None)
        self.parent = parent
        self.score = score          # このノードの手番のプレイヤーから見た評価値 (終局なら終局の評価値)
        self.terminal = terminal
        self.children = []          # 展開した場合の子ノード (全ての合法手)
        self.state = None           # 展開するノードの局面 (展開後は不要なので破棄する)

    def value(self) -> float:
        """このノードの手番のプレイヤーから見た値 (展開した子があればミニマックス法で求める)"""
        if not self.children:
            return self.score
        return max(-child.value() for child in self.children)


def _expand(node: _Node, ply: int) -> list:
    """node の全ての合法手を指して採点し, 子ノードのリストを返す"""
    state = node.state
    for move in state.get_legal_moves().tolist():
        state.play(move)
        if state.is_terminal:
            child = _Node(move, node, _terminal_score(state, ply + 1), True)
        else:
            child = _Node(move, node, evaluate(state), False)
        state.undo()
        node.children.append(child)
    node.state = None
    return node.children


def _beam_search_root(state, depth: int, beam_width: int):
    """state の手番のプレイヤーの最善手と評価値を, ビームサーチで求める"""
    root = _Node(None, None, 0.0, False)
    root.state = state
    frontier = [root]
    for level in range(depth):
        children = []
        for node in frontier:
            children.extend(_expand(node, level))
        if level == depth - 1:
            break
        # その手を指したプレイヤーから見た評価値 (= -子の手番から見た評価値) が高い順に beam_width 個を残す
        candidates = [c for c in children if not c.terminal]
        candidates.sort(key=lambda c: -c.score, reverse=True)
        frontier = candidates[:beam_width]
        for child in frontier:
            # 親の局面から1手進めた局面を作る (親の局面は展開時に破棄済みなので, 根から手順を再生する)
            path = []
            node = child
            while node.parent is not None:
                path.append(node.move)
                node = node.parent
            child.state = state.copy()
            for move in reversed(path):
                child.state.play(move)
        if not frontier:
            break

    # 根の子を, 1手先の評価値の高い順 (agent_ab.py の手の並べ替えと同じ順) に並べてから最大の手を選ぶ
    ordered = sorted(root.children, key=lambda c: -c.score, reverse=True)
    best = max(ordered, key=lambda c: -c.value())
    return best.move, -best.value()


def beam_move(obs: pybl.Observation, depth: int | None = None, num_samples: int | None = None,
              beam_width: int | None = None, endgame_depth: int | None = None, rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, 着手をビームサーチで選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.
    depth: 山札が残っている間の探索の深さ (1以上. 省略時は DEPTH)
    num_samples: 探索開始時に生成する局面の数 (1以上. 省略時は NUM_SAMPLES)
    beam_width: 各深さで次の深さに進める (展開する) ノードの数 (1以上. 省略時は BEAM_WIDTH)
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
