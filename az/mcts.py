"""ネットワークを使う ISMCTS (AlphaZero 型の探索).

v1 の ISMCTS (agent_ismcts.py) と同じく, シミュレーションごとに自分から見えないカード (相手の手札と山札) を
配り直した局面 (determinization) を使う. AlphaZero と同じく, ロールアウトの代わりにネットワークの価値で
葉を評価し, ネットワークの方策を事前確率として PUCT で手を選ぶ.

1シミュレーションの流れ:
    determinization → 選択 (PUCT) → 新しいノードに来たらネットワークで評価して終わる → 逆伝播
    (終局した局面では, ネットワークの代わりに勝敗 (勝ち +1・引き分け 0・負け -1) を使う)

選択 (PUCT): 子 a の値 = Q(a) + c_puct × P(a) × √(Σ_b N(b)) / (1 + N(a))
    Q(a): 子 a の平均価値 (その手を指したプレイヤーから見た値, -1〜1). 未訪問なら fpu
    P(a): ノードを評価したときに保存した方策のロジットを, この determinization で合法な手だけで softmax したもの
    Σ_b N(b): この determinization で合法な手の子の訪問回数の合計 (ISMCTS の availability に当たる)

バッチ推論: batch_size 個のシミュレーションを先に進め, 葉をまとめてネットワークで評価する.
    同じ葉にシミュレーションが集中しないよう, 評価を待つ間は仮想損失 (N を 1 増やし, 負けとして数える) を加える.

ルート:
    - v1 と同じく, 厳密に同等な手 (フラッグの左右対称・色の入れ替え) を1つの代表にまとめて探索する
    - 自己対戦では, 事前確率にディリクレノイズを混ぜて探索を促す (dirichlet_alpha > 0)
    - 結果の訪問回数の分布は, 代表の手の訪問回数を同等な手に均等に分けて, 541 通りの行動に戻す (学習の正解 π)
"""
import math
import random
from dataclasses import dataclass

import numpy as np

import pybl
from az import features as F


class Node:
    __slots__ = ("move", "player", "parent", "children", "visits", "value_sum", "logits")

    def __init__(self, move, player, parent):
        self.move = move            # このノードに至った着手 (ルートは None)
        self.player = player        # 手を指したプレイヤー (value_sum はこの視点. ルートは自分の相手)
        self.parent = parent
        self.children = {}          # 着手 -> 子ノード
        self.visits = 0             # 訪問回数 N
        self.value_sum = 0.0        # 価値の合計 W (-1〜1 の値の合計)
        self.logits = None          # ネットワークで評価したときの方策のロジット (541,). 未評価なら None


class NetworkEvaluator:
    """ネットワークで局面をまとめて評価する: (planes, scalars) → (方策のロジット (B, 541), 価値 (B,))"""

    def __init__(self, model, device="cpu"):
        import torch
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, planes: np.ndarray, scalars: np.ndarray) -> tuple:
        torch = self.torch
        with torch.no_grad():
            p = torch.as_tensor(planes, dtype=torch.float32, device=self.device)
            s = torch.as_tensor(scalars, dtype=torch.float32, device=self.device)
            logits, value = self.model(p, s)
        return logits.float().cpu().numpy(), value.float().cpu().numpy()


@dataclass
class SearchResult:
    moves: list             # ルートで探索した手 (代表の手)
    visits: np.ndarray      # 各手の訪問回数
    q_values: np.ndarray    # 各手の平均価値 (自分から見た値, -1〜1. 未訪問は 0)
    policy: np.ndarray      # 訪問回数の分布を 541 通りの行動に戻したもの (同等な手には均等に分ける)
    root_value: float       # ルートの価値の推定 (自分から見た値)

    def best_move(self) -> int:
        """訪問回数が最大の手 (同数なら平均価値が大きい手)"""
        order = np.lexsort((self.q_values, self.visits))
        return int(self.moves[order[-1]])

    def sample_move(self, temperature: float, rng: random.Random) -> int:
        """訪問回数^(1/temperature) に比例した確率で手を選ぶ (temperature <= 0 なら best_move)"""
        if temperature <= 0:
            return self.best_move()
        weights = self.visits.astype(np.float64) ** (1.0 / temperature)
        if weights.sum() <= 0:
            return self.best_move()
        return int(self.moves[rng.choices(range(len(self.moves)), weights=weights)[0]])


def _equivalence_classes(obs: pybl.Observation, use_representatives: bool) -> dict:
    """ルートの代表の手 -> それと同等な合法手のリスト"""
    legal = obs.get_legal_moves().tolist()
    if not use_representatives:
        return {m: [m] for m in legal}
    from agent_ismcts import _color_representatives, _flag_representatives
    flag_rep, color_rep = _flag_representatives(obs), _color_representatives(obs)
    classes = {}
    for move in legal:
        rep = move
        if move != pybl.PASS_MOVE:
            card = pybl.move_card(move)
            rep_card = pybl.make_card(color_rep[int(pybl.card_color(card))], pybl.card_rank(card))
            rep = int(pybl.make_move(rep_card, flag_rep[pybl.move_flag(move)]))
        classes.setdefault(rep, []).append(move)
    return classes


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _terminal_value(state, player) -> float:
    """終局した局面の, player から見た価値"""
    winner = state.winner
    if winner == pybl.NULL_PLAYER:
        return 0.0
    return 1.0 if winner == player else -1.0


def search(obs: pybl.Observation, evaluator, num_simulations: int = 200, c_puct: float = 1.5,
           batch_size: int = 16, fpu: float = 0.0, dirichlet_alpha: float = 0.0, dirichlet_epsilon: float = 0.25,
           use_representatives: bool = True, rng: random.Random | None = None) -> SearchResult:
    """obs (手番のプレイヤーの観測) から ISMCTS で探索し, ルートの統計を返す.

    evaluator: (planes, scalars) をまとめて受け取り (方策のロジット, 価値) を返す関数 (NetworkEvaluator など)
    num_simulations: シミュレーションの回数
    c_puct: PUCT の探索係数
    batch_size: まとめてネットワークで評価する葉の数
    fpu: 未訪問の子の Q の値 (first play urgency)
    dirichlet_alpha: ルートの事前確率に混ぜるディリクレノイズの集中度 (0 なら混ぜない)
    dirichlet_epsilon: ディリクレノイズを混ぜる割合
    use_representatives: ルートで同等な手を1つの代表にまとめるか
    rng: 乱数 (determinization, ノイズ)
    """
    rng = rng or random.Random()
    me = obs.player
    classes = _equivalence_classes(obs, use_representatives)
    root_moves = list(classes)
    root = Node(None, pybl.to_opponent(me), None)

    # ルートの事前確率 (代表の手には, 同等な手の確率の合計を割り当てる)
    planes, scalars = F.encode(obs)
    logits, values = evaluator(planes[None], scalars[None])
    legal = obs.get_legal_moves()
    probs = dict(zip(legal.tolist(), _softmax(logits[0][legal])))
    root_prior = np.array([sum(probs[m] for m in classes[r]) for r in root_moves])
    if dirichlet_alpha > 0 and len(root_moves) > 1:
        noise = np.random.default_rng(rng.getrandbits(64)).dirichlet([dirichlet_alpha] * len(root_moves))
        root_prior = (1 - dirichlet_epsilon) * root_prior + dirichlet_epsilon * noise
    root_prior_of = dict(zip(root_moves, root_prior))
    root_value = float(values[0])

    base = obs.sample_state(seed=rng.getrandbits(64))
    done = 0
    while done < num_simulations:
        pending = []        # (葉のノード, 経路, 葉の局面の手番のプレイヤー)
        inputs = []
        for _ in range(min(batch_size, num_simulations - done)):
            done += 1
            work = base.copy()
            work.determinize(me, seed=rng.getrandbits(64))
            node, path = root, []
            while True:
                if work.is_terminal:
                    _backup(path, lambda p, s=work: _terminal_value(s, p), virtual=True)
                    break
                if node is not root and node.logits is None:
                    if node.visits > 1:     # 同じバッチの別のシミュレーションが評価待ちにしたノード
                        _backup(path, lambda p: 0.0, virtual=True)
                        break
                    o = work.observe(work.side_to_move)
                    planes, scalars = F.encode(o)
                    inputs.append((planes, scalars))
                    pending.append((node, path, work.side_to_move))
                    break
                move = _select(node, root, root_moves, root_prior_of, work, c_puct, fpu)
                child = node.children.get(move)
                if child is None:
                    child = Node(move, work.side_to_move, node)
                    node.children[move] = child
                work.play(move)
                node = child
                path.append(node)
                node.visits += 1            # 仮想損失: 評価を待つ間は負けとして数える
                node.value_sum -= 1.0

        if pending:
            logits, values = evaluator(np.stack([x[0] for x in inputs]), np.stack([x[1] for x in inputs]))
            for (node, path, to_move), node_logits, value in zip(pending, logits, values):
                if node.logits is None:
                    node.logits = node_logits
                v = float(value)
                _backup(path, lambda p, v=v, t=to_move: v if p == t else -v, virtual=True)

    children = [root.children.get(m) for m in root_moves]
    visits = np.array([c.visits if c else 0 for c in children], dtype=np.int64)
    q_values = np.array([c.value_sum / c.visits if c and c.visits else 0.0 for c in children])
    policy = np.zeros(F.NUM_ACTIONS, dtype=np.float32)
    total = visits.sum()
    if total > 0:
        for rep, n in zip(root_moves, visits):
            for m in classes[rep]:
                policy[m] = n / total / len(classes[rep])
    if total > 0:
        root_value = float((visits * q_values).sum() / total)
    return SearchResult(root_moves, visits, q_values, policy, root_value)


def _select(node: Node, root: Node, root_moves: list, root_prior_of: dict, work, c_puct: float, fpu: float) -> int:
    """PUCT で手を選ぶ"""
    if node is root:
        legal = root_moves
        priors = [root_prior_of[m] for m in legal]
    else:
        legal = work.get_legal_moves()
        priors = _softmax(node.logits[legal])
        legal = legal.tolist()
    children = node.children
    total = 0
    for m in legal:
        child = children.get(m)
        if child is not None:
            total += child.visits
    sqrt_total = math.sqrt(max(total, 1))
    best, best_score = legal[0], -math.inf
    for m, p in zip(legal, priors):
        child = children.get(m)
        if child is None or child.visits == 0:
            score = fpu + c_puct * p * sqrt_total
        else:
            score = child.value_sum / child.visits + c_puct * p * sqrt_total / (1 + child.visits)
        if score > best_score:
            best, best_score = m, score
    return best


def _backup(path: list, value_for, virtual: bool) -> None:
    """経路上の各ノードに, そのノードの手を指したプレイヤーから見た価値を加える (仮想損失を取り除く)"""
    for node in path:
        if virtual:
            node.value_sum += 1.0       # 仮想損失 (-1) を取り除く (訪問回数はそのまま1回として数える)
        node.value_sum += value_for(node.player)
