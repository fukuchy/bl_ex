"""ロールアウトを「打ち切り＋静的評価」に置き換えた ISMCTS のエージェント.

v1 の ISMCTS (agent_ismcts.py, doc/ismcts_spec.md) を基にしている. 選択 (availability count 付き UCB1),
1反復1ノードの展開, 逆伝播, ルートの同等手のまとめ, 着手の決定は v1 と同じで, ロールアウトだけが異なる.

1反復の流れ:
    determinization → 選択 → 展開 → rollout_depth 手だけランダムに進める → 評価関数で採点して逆伝播
    (途中で終局した場合は, v1 と同じく勝ち 1・引き分け 0.5・負け 0 を報酬とする)

評価関数 (evaluator):
    "ab": agent_ab.py の評価関数 (確保フラッグ数の差 + 各フラッグで作れる最強の役の差) を,
          シグモイド関数 1 / (1 + exp(-評価値 / scale)) で 0〜1 の勝ちやすさに変換する
    "dp": agent_dp.py の確率モデルが求める「ゲーム全体で勝つ確率」(0〜1) をそのまま使う (ab より重い)

評価関数は手番のプレイヤーから見た値を返すので, 各ノードの報酬は「そのノードの手を指したプレイヤー」から見た値
(手番のプレイヤーなら v, 相手なら 1 - v) に直して逆伝播する.

答えを見た版 (oracle_eval_move):
    determinization をせず, 相手の手札と山札を含む本物の局面で探索する (agent_oracle.py と同じ考え方).
    評価を改善したときの, 隠れ情報の扱いの伸びしろを測るための比較用.

実行時の指定 (match.py / play_vs_agent.py):
    python3 match.py 400 ismcts_eval@num_iterations=3000 ismcts@num_iterations=3000 -j 8
    python3 match.py 400 ismcts_eval@evaluator=dp,num_iterations=1000 abdeck -j 8
    python3 match.py 400 ismcts_eval@rollout_depth=2,scale=50,c=0.3 abdeck -j 8
    python3 match.py 400 oracle_eval@num_iterations=3000 ismcts_eval@num_iterations=3000 -j 8

使い方:
    import pybl
    from agent_ismcts_eval import ismcts_eval_move

    state = pybl.GameState()
    move = ismcts_eval_move(state.observe(state.side_to_move), num_iterations=3000)
    state.play(move)
"""
import math
import random

import pybl

import agent_ab
import agent_dp
import agent_oracle
from agent_ismcts import EXPLORATION, NUM_ITERATIONS, Node, _reward, representative_moves

EVALUATOR = "ab"        # 評価関数 ("ab" または "dp")
ROLLOUT_DEPTH = 0       # 評価する前にランダムに進める手数 (0 なら展開したノードをその場で評価する)
SCALE = 100.0           # ab の評価値を 0〜1 に変換するシグモイド関数の尺度 (フラッグ1本の差 = 評価値 100)

_default_rng = random.Random()


def _ab_value(state, scale: float) -> float:
    """ab の評価関数による, 手番のプレイヤーから見た勝ちやすさ (0〜1)"""
    x = agent_ab.evaluate(state) / scale
    x = max(-50.0, min(50.0, x))        # exp のオーバーフローを避ける
    return 1.0 / (1.0 + math.exp(-x))


def _dp_value(state, scale: float) -> float:
    """dp の確率モデルによる, 手番のプレイヤーから見たゲーム全体で勝つ確率 (0〜1)"""
    obs = state.observe(state.side_to_move)
    flags, owners = agent_dp._flags_of(obs, obs.player)
    return agent_dp._Evaluator(obs).evaluate(flags, owners, int(obs.hand))


_EVALUATORS = {"ab": _ab_value, "dp": _dp_value}


def _leaf_rewards(work, rollout_depth: int, evaluate, scale: float, rng: random.Random):
    """展開したノードの局面を採点し, ノードの手を指したプレイヤーから見た報酬を求める関数を返す"""
    for _ in range(rollout_depth):
        if work.is_terminal:
            break
        moves = work.get_legal_moves()
        work.play(moves[rng.randrange(len(moves))])

    if work.is_terminal:
        result = work.result
        return lambda player: _reward(result, player)

    to_move = work.side_to_move
    v = evaluate(work, scale)
    return lambda player: v if player == to_move else 1.0 - v


def _search(base, me, root_moves: list, num_iterations: int, c: float, rollout_depth: int,
            evaluator: str, scale: float, rng: random.Random, determinize: bool) -> Node:
    """ISMCTS (determinize=False なら本物の局面での MCTS) で探索し, ルートノードを返す"""
    evaluate = _EVALUATORS[evaluator]
    root = Node(None, pybl.to_opponent(me), None)
    work = pybl.GameState(seed=0)

    for _ in range(num_iterations):
        # 1. determinization: 基準の局面を複製し, 自分から見えないカードを配り直す
        base.copy_to(work)
        if determinize:
            work.determinize(me, seed=rng.getrandbits(64))

        # 2. 選択 / 3. 展開 (v1 と同じ)
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

        # 4. 打ち切り＋静的評価 (v1 ではランダムプレイアウト)
        reward = _leaf_rewards(work, rollout_depth, evaluate, scale, rng)

        # 5. 逆伝播
        while node is not None:
            node.visits += 1
            node.reward += reward(node.player)
            node = node.parent

    return root


def _check_params(num_iterations, c, rollout_depth, evaluator, scale):
    if evaluator not in _EVALUATORS:
        raise ValueError(f"evaluator must be one of {', '.join(_EVALUATORS)}: {evaluator}")
    if int(num_iterations) < 1 or int(rollout_depth) < 0 or float(scale) <= 0 or float(c) < 0:
        raise ValueError("num_iterations >= 1, rollout_depth >= 0, scale > 0 and c >= 0 are required")


def _choose(root: Node):
    """訪問回数が最大の手. 同数なら平均報酬が大きい手 (v1 と同じ)"""
    best = max(root.children.values(), key=lambda n: (n.visits, n.reward / n.visits))
    return pybl.Move(best.move)


def _check_obs(obs):
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    if len(obs.get_legal_moves()) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")


def ismcts_eval_move(obs: pybl.Observation, num_iterations: int = NUM_ITERATIONS, c: float = EXPLORATION,
                     rollout_depth: int = ROLLOUT_DEPTH, evaluator: str = EVALUATOR, scale: float = SCALE,
                     rng: random.Random | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, 打ち切り＋静的評価の ISMCTS で着手を選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.
    num_iterations: 1手あたりの反復回数 (既定値は v1 の agent_ismcts.NUM_ITERATIONS と同じ)
    c: UCB1 の探索係数 (既定値は v1 の agent_ismcts.EXPLORATION と同じ)
    rollout_depth: 評価する前にランダムに進める手数 (0 なら展開したノードをその場で評価する)
    evaluator: 評価関数 ("ab" または "dp")
    scale: ab の評価値を 0〜1 に変換するシグモイド関数の尺度 (evaluator="ab" のときのみ使う)
    rng: random.Random を渡すと, 探索の全ての乱数をそこから作る (同じシードなら同じ手を返す)

    置ける場所が無い場合は pybl.PASS_MOVE を返す. 終局している場合は ValueError を送出する.
    """
    _check_obs(obs)
    _check_params(num_iterations, c, rollout_depth, evaluator, scale)
    rng = rng or _default_rng

    root_moves = representative_moves(obs)
    if len(root_moves) == 1:
        return pybl.Move(root_moves[0])

    base = obs.sample_state(seed=rng.getrandbits(64))   # 基準の局面
    root = _search(base, obs.player, root_moves, int(num_iterations), float(c), int(rollout_depth),
                   evaluator, float(scale), rng, determinize=True)
    return _choose(root)


def oracle_eval_move(obs: pybl.Observation, state: pybl.GameState | None = None,
                     num_iterations: int = NUM_ITERATIONS, c: float = EXPLORATION,
                     rollout_depth: int = ROLLOUT_DEPTH, evaluator: str = EVALUATOR, scale: float = SCALE,
                     rng: random.Random | None = None):
    """相手の手札と山札を含む本物の局面 state を使い, 打ち切り＋静的評価の MCTS で着手を選んで返す (比較用).

    determinization をしない点以外は ismcts_eval_move と同じ. state は match.py / play_vs_agent.py が自動で渡す.
    """
    _check_obs(obs)
    if not isinstance(state, pybl.GameState):
        raise TypeError("oracle_eval_move requires the true game state (state=pybl.GameState)")
    if state.observe(obs.player) != obs:
        raise ValueError("state does not match obs")
    _check_params(num_iterations, c, rollout_depth, evaluator, scale)
    rng = rng or _default_rng

    root_moves = agent_oracle.representative_moves(obs, state)
    if len(root_moves) == 1:
        return pybl.Move(root_moves[0])

    root = _search(state, obs.player, root_moves, int(num_iterations), float(c), int(rollout_depth),
                   evaluator, float(scale), rng, determinize=False)
    return _choose(root)
