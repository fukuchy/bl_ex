"""相手の手札と山札を見た「答えを見た版」の完全情報 MCTS エージェント (比較実験用).

v1 の ISMCTS (agent_ismcts.py, doc/ismcts_spec.md) と同じ仕様で, determinization をしない点だけが異なる.
反復ごとに見えないカードを配り直す代わりに, 相手の手札と山札の順番を含む本物の局面をそのまま使って探索する.
隠れ情報の扱い (情報漏れの解消と相手の手札の推論) を完璧にしたときの改善の上限を測るためのもので,
実際の対局では使えない (ルール上見えない情報を使う).

v1 と同じ部分: ノードの構造, availability count 付き UCB1 による選択, 1反復1ノードの展開,
PyBL のランダムプレイアウト, 報酬 (勝ち 1・引き分け 0.5・負け 0), 訪問回数が最大の手の選択,
反復回数と探索係数 c の既定値 (agent_ismcts.py の NUM_ITERATIONS, EXPLORATION を使う).

ルートの同等手のまとめは, 本物の局面でも厳密に成り立つものに限る.
- フラッグの左右対称: 盤面だけで決まるので v1 と同じ
- 色の入れ替え: 相手の手札と山札の中身まで一致する場合に限る. 山札の順番は区別するため,
  山札にカードがある色は他の色とまとめない

match.py / play_vs_agent.py は, 関数が state 引数を受け取る場合に本物の局面を渡す:
    python3 match.py 400 oracle ismcts -j 8

使い方:
    import pybl
    from agent_oracle import oracle_move

    state = pybl.GameState()
    move = oracle_move(state.observe(state.side_to_move), state=state)
    state.play(move)
"""
import random

import pybl

from agent_ismcts import EXPLORATION, NUM_ITERATIONS, Node, _flag_representatives, _reward

_default_rng = random.Random()


def _card_locations(state: pybl.GameState, me) -> list:
    """本物の局面での各カードの所在. 山札のカードは順番を区別するため, カードごとに別の所在とする"""
    opp = pybl.to_opponent(me)
    locations = [None] * pybl.NUM_CARDS
    for card in pybl.CardIterator(state.get_hand(me)):
        locations[int(card)] = "hand"
    for card in pybl.CardIterator(state.get_hand(opp)):
        locations[int(card)] = "opponent_hand"
    for flag in range(pybl.NUM_FLAGS):
        for player in (pybl.FIRST, pybl.SECOND):
            for card in state.get_flag_cards(flag, player):
                locations[int(card)] = (flag, player == me)
    for card in range(pybl.NUM_CARDS):
        if locations[card] is None:
            locations[card] = ("deck", card)
    return locations


def representative_moves(obs: pybl.Observation, state: pybl.GameState) -> list:
    """本物の局面で厳密に同等な手を1つの代表にまとめた合法手 (代表の手は全て合法手)"""
    flag_rep = _flag_representatives(obs)
    locations = _card_locations(state, obs.player)
    pattern_to_color = {}
    color_rep = []
    for color in range(pybl.NUM_COLORS):
        pattern = tuple(locations[int(pybl.make_card(color, rank))] for rank in range(1, pybl.NUM_RANKS + 1))
        color_rep.append(pattern_to_color.setdefault(pattern, color))

    representatives, seen = [], set()
    for move in obs.get_legal_moves().tolist():
        if move != pybl.PASS_MOVE:
            card = pybl.move_card(move)
            rep_card = pybl.make_card(color_rep[int(pybl.card_color(card))], pybl.card_rank(card))
            move = int(pybl.make_move(rep_card, flag_rep[pybl.move_flag(move)]))
        if move not in seen:
            seen.add(move)
            representatives.append(move)
    return representatives


def _search(state: pybl.GameState, me, root_moves: list, num_iterations: int, c: float, rng: random.Random) -> Node:
    """本物の局面で MCTS を行い, ルートノードを返す (v1 の _search から determinization を除いたもの)"""
    root = Node(None, pybl.to_opponent(me), None)
    work = pybl.GameState(seed=0)

    for _ in range(num_iterations):
        # determinization はしない: 本物の局面 (相手の手札と山札の順番を含む) をそのまま使う
        state.copy_to(work)

        # 選択 / 展開
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

        # ロールアウト
        result = work.result if work.is_terminal else work.random_playout(seed=rng.getrandbits(64))

        # 逆伝播
        while node is not None:
            node.visits += 1
            node.reward += _reward(result, node.player)
            node = node.parent

    return root


def oracle_move(obs: pybl.Observation, state: pybl.GameState | None = None, num_iterations: int = NUM_ITERATIONS,
                c: float = EXPLORATION, rng: random.Random | None = None):
    """相手の手札と山札を含む本物の局面 state を使い, 完全情報 MCTS で着手を選んで返す.

    obs: 手番のプレイヤーの pybl.Observation
    state: 本物の局面 (pybl.GameState). match.py / play_vs_agent.py が自動で渡す
    num_iterations: 1手あたりの反復回数 (既定値は v1 の agent_ismcts.NUM_ITERATIONS と同じ)
    c: UCB1 の探索係数 (既定値は v1 の agent_ismcts.EXPLORATION と同じ)
    rng: random.Random を渡すと, 探索の全ての乱数をそこから作る (同じシードなら同じ手を返す)

    置ける場所が無い場合は pybl.PASS_MOVE を返す. 終局している場合は ValueError を送出する.
    """
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    if not isinstance(state, pybl.GameState):
        raise TypeError("oracle_move requires the true game state (state=pybl.GameState)")
    if state.observe(obs.player) != obs:
        raise ValueError("state does not match obs")
    rng = rng or _default_rng
    if len(obs.get_legal_moves()) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")

    root_moves = representative_moves(obs, state)
    if len(root_moves) == 1:
        return pybl.Move(root_moves[0])

    root = _search(state, obs.player, root_moves, num_iterations, c, rng)

    # 訪問回数が最大の手. 同数なら平均報酬が大きい手 (v1 と同じ)
    best = max(root.children.values(), key=lambda n: (n.visits, n.reward / n.visits))
    return pybl.Move(best.move)
