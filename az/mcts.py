"""ネットワークを使う ISMCTS (AlphaZero 型の探索).

v1 の ISMCTS (agent_ismcts.py) と同じく, シミュレーションごとに自分から見えないカード (相手の手札と山札) を
配り直した局面 (determinization) を使う. AlphaZero と同じく, ロールアウトの代わりにネットワークの価値で
葉を評価し, ネットワークの方策を事前確率として PUCT で手を選ぶ.

1シミュレーションの流れ:
    determinization → 選択 (PUCT) → 新しいノードに来たらネットワークで評価して終わる → 逆伝播
    (終局した局面では, ネットワークの代わりに勝敗 (勝ち +1・引き分け 0・負け -1) を使う)

選択 (PUCT): 子 a の値 = Q(a) + c_puct × P(a) × √(Σ_b N(b)) / (1 + N(a))
    Q(a): 子 a の平均価値 (その手を指したプレイヤーから見た値, -1〜1). 未訪問なら fpu
          (fpu_reduction を指定すると, 未訪問の手の Q を 訪問済みの子の平均価値 − fpu_reduction × √(訪問済みの子の P の合計)
           にする (Leela Zero の FPU reduction). 負けている局面で未訪問の手ばかりが選ばれて探索が広がりすぎるのを防ぐ)
    P(a): ノードを評価したときに保存した方策のロジットを, この determinization で合法な手だけで softmax したもの
    Σ_b N(b): この determinization で合法な手の子の訪問回数の合計 (ISMCTS の availability に当たる)

バッチ推論: batch_size 個のシミュレーションを先に進め, 葉をまとめてネットワークで評価する.
    search_steps は評価をジェネレータの外に任せるので, 複数の探索 (自己対戦の複数の局) の葉をまとめて GPU で評価できる.
    同じ葉にシミュレーションが集中しないよう, 評価を待つ間は仮想損失 (N を 1 増やし, 負けとして数える) を加える.

ルート:
    - v1 と同じく, 厳密に同等な手 (フラッグの左右対称・色の入れ替え) を1つの代表にまとめて探索する
    - 自己対戦では, 事前確率にディリクレノイズを混ぜて探索を促す (dirichlet_alpha > 0)
    - 結果の訪問回数の分布は, 代表の手の訪問回数を同等な手に均等に分けて, 541 通りの行動に戻す (学習の正解 π)

ルート以外のノードの事前確率 (prior_cache):
    none (既定): ノードを最初に評価したときの determinization でのロジットを, 以降の別の determinization でも使い回す.
          ルート以外では手番のプレイヤーの手札が determinization ごとに違う (相手のノードでは相手の手札,
          深さ 2 以降の自分のノードでは山札から引いたカード) ので, 評価したときの手札に無かったカードの手のロジット
          (学習で勾配が流れていない値) で PUCT が動くことがある
    hand: ノードごとに {手番のプレイヤーの手札: ロジット} を持ち, 今の determinization の手札のロジットを使う.
          まだ評価していない手札に当たったら, そのノードをこのシミュレーションの葉として今の局面をネットワークで評価し,
          ロジットをその手札で保存して価値を逆伝播する (1シミュレーションあたりの評価は1回のまま).
          同じバッチの別のシミュレーションが評価待ちにしたノードと手札でも, 価値 0 で済ませずに評価を要求する
          (同じ入力を2回評価することになるが, ロジットは先に保存したものを使う)
    stats (PriorStats) を渡すと, 事前確率の使い回しの度合いを集計する (診断用. az.diag_prior)
"""
import math
import random
from dataclasses import dataclass

import numpy as np

import pybl
from az import features as F


class _PyNode:
    __slots__ = ("move", "player", "parent", "children", "visits", "value_sum", "logits", "hand", "cache")

    def __init__(self, move, player, parent):
        self.move = move            # このノードに至った着手 (ルートは None)
        self.player = player        # 手を指したプレイヤー (value_sum はこの視点. ルートは自分の相手)
        self.parent = parent
        self.children = {}          # 着手 -> 子ノード
        self.visits = 0             # 訪問回数 N
        self.value_sum = 0.0        # 価値の合計 W (-1〜1 の値の合計)
        self.logits = None          # ネットワークで評価したときの方策のロジット (541,). 未評価なら None
        self.hand = None            # logits を評価したときの手番のプレイヤーの手札 (stats を集計するときだけ記録する)
        self.cache = None           # prior_cache="hand" のときの {手番のプレイヤーの手札: ロジット}


try:
    from az._fast import Node      # Cython 版 (python setup.py build_ext --inplace でビルドする)
except ImportError:
    Node = _PyNode
if Node is not _PyNode and not hasattr(Node(None, None, None), "cache"):
    raise ImportError("az/_fast (Cython 版) のビルドが古い. bl_ex のディレクトリで python setup.py build_ext --inplace を実行する")


class NetworkEvaluator:
    """ネットワークで局面をまとめて評価する: (planes, scalars) → (方策のロジット (B, 541), 価値 (B,))"""

    def __init__(self, model, device="cpu", half: bool | None = None):
        """half: 半精度 (float16) で推論するか. 省略時は GPU (cuda) なら半精度にする"""
        import torch
        self.torch = torch
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.half = self.device.type == "cuda" if half is None else half

    def __call__(self, planes: np.ndarray, scalars: np.ndarray) -> tuple:
        torch = self.torch
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.half):
            p = torch.as_tensor(planes, device=self.device).float()     # uint8 のまま転送してから変換する
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


PRIOR_CACHES = ("none", "hand")


class PriorStats:
    """診断用: 探索中に, ルート以外のノードで事前確率をどれだけ別の手札のものから使い回しているかを集計する.

    search / search_steps に stats として渡す (渡さなければ集計しないので, 通常の探索は遅くならない).
    深さはルートからの手数 (1, 2, 3 以上にまとめる), 手番はルートのプレイヤーか相手かで分ける.

    select[(深さ, ルートのプレイヤーの手番か)] = [選択の回数, 手札が違った回数, 無いカードの手の割合の合計,
                                               無いカードの手の子の訪問回数の合計, 合法手の子の訪問回数の合計]
        手札が違う: 今の determinization の手番のプレイヤーの手札が, 使ったロジットを評価したときの手札と違う
        無いカードの手: 今の合法手のうち, 評価したときの手札に無かったカードを使う手
    arrive[(深さ, ルートのプレイヤーの手番か)] = [到達の回数, 手札のロジットがまだ無かった回数] (prior_cache="hand")
    leaves: 葉の種類ごとの数 (eval: ネットワークで評価, terminal: 終局, collision: 評価待ちと重なって価値 0) と深さの合計
    """

    def __init__(self):
        self.searches = 0
        self.select = {}
        self.arrive = {}
        self.leaves = {"eval": 0, "terminal": 0, "collision": 0, "depth": 0}

    def record_leaf(self, kind: str, depth: int) -> None:
        self.leaves[kind] += 1
        self.leaves["depth"] += depth

    def record_arrive(self, depth: int, own: bool, miss: bool) -> None:
        row = self.arrive.setdefault((min(depth, 3), own), [0, 0])
        row[0] += 1
        row[1] += miss

    def record_select(self, depth: int, own: bool, saved_hand: int, work, node) -> None:
        """node で選択を行う直前に呼ぶ. saved_hand は使うロジットを評価したときの手番のプレイヤーの手札"""
        current = int(work.get_hand(work.side_to_move))
        missing = current & ~saved_hand
        legal = work.get_legal_moves().tolist()
        count, weighted, total = 0, 0, 0
        for m in legal:
            child = node.children.get(m)
            n = child.visits if child is not None else 0
            total += n
            if m != _PASS and (missing >> (m // F.NUM_FLAGS)) & 1:
                count += 1
                weighted += n
        row = self.select.setdefault((min(depth, 3), own), [0, 0, 0.0, 0, 0])
        row[0] += 1
        row[1] += current != saved_hand
        row[2] += count / len(legal)
        row[3] += weighted
        row[4] += total


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


_PASS = int(pybl.PASS_MOVE)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _terminal_value(state, player) -> float:
    """終局した局面の, player から見た価値"""
    winner = state.winner
    if winner == pybl.NULL_PLAYER:
        return 0.0
    return 1.0 if winner == player else -1.0


def run_steps(steps, evaluator):
    """評価の要求を出すジェネレータ (search_steps など) を, evaluator で評価しながら最後まで進めて, その戻り値を返す"""
    try:
        request = next(steps)
        while True:
            request = steps.send(evaluator(*request))
    except StopIteration as stop:
        return stop.value


def search(obs: pybl.Observation, evaluator, num_simulations: int = 200, c_puct: float = 1.5,
           batch_size: int = 16, fpu: float = 0.0, fpu_reduction: float | None = None, dirichlet_alpha: float = 0.0,
           dirichlet_epsilon: float = 0.25, use_representatives: bool = True, sample_world=None,
           prior_cache: str = "none", stats: PriorStats | None = None,
           rng: random.Random | None = None) -> SearchResult:
    """obs (手番のプレイヤーの観測) から ISMCTS で探索し, ルートの統計を返す.

    evaluator: (planes, scalars) をまとめて受け取り (方策のロジット, 価値) を返す関数 (NetworkEvaluator など)
    その他の引数は search_steps と同じ
    """
    return run_steps(search_steps(obs, num_simulations=num_simulations, c_puct=c_puct, batch_size=batch_size, fpu=fpu,
                                  fpu_reduction=fpu_reduction, dirichlet_alpha=dirichlet_alpha,
                                  dirichlet_epsilon=dirichlet_epsilon, use_representatives=use_representatives,
                                  sample_world=sample_world, prior_cache=prior_cache, stats=stats, rng=rng),
                     evaluator)


def search_steps(obs: pybl.Observation, num_simulations: int = 200, c_puct: float = 1.5, batch_size: int = 16,
                 fpu: float = 0.0, fpu_reduction: float | None = None, dirichlet_alpha: float = 0.0,
                 dirichlet_epsilon: float = 0.25, use_representatives: bool = True, sample_world=None,
                 prior_cache: str = "none", stats: PriorStats | None = None, rng: random.Random | None = None):
    """search と同じ探索を, ネットワークの評価を外に任せるジェネレータとして行う.

    評価してほしい局面を (planes (B, ...), scalars (B, ...)) として yield し, (方策のロジット (B, 541), 価値 (B,)) を
    send で受け取る. 終わると SearchResult を返す (StopIteration.value). 複数の探索の要求をまとめて評価するのに使う.

    num_simulations: シミュレーションの回数
    c_puct: PUCT の探索係数
    batch_size: まとめてネットワークで評価する葉の数
    fpu: 未訪問の子の Q の値 (first play urgency). fpu_reduction を指定した場合は使わない
    fpu_reduction: 指定すると, 未訪問の子の Q を 訪問済みの子の平均価値 − fpu_reduction × √(訪問済みの子の P の合計) にする
    dirichlet_alpha: ルートの事前確率に混ぜるディリクレノイズの集中度 (0 なら混ぜない)
    dirichlet_epsilon: ディリクレノイズを混ぜる割合
    use_representatives: ルートで同等な手を1つの代表にまとめるか
    sample_world: シミュレーションごとの局面を作る関数 (rng → GameState). 省略時は見えないカードを配り直す
                  (determinization). 診断用に, 本当の相手の手札を使う局面などを渡せる (revealed_world_sampler)
    prior_cache: ルート以外のノードの事前確率の扱い. none (評価したときのロジットを使い回す) / hand (手札ごとに評価する)
    stats: PriorStats を渡すと, 事前確率の使い回しの度合いを集計する (診断用)
    rng: 乱数 (determinization, ノイズ)
    """
    if prior_cache not in PRIOR_CACHES:
        raise ValueError(f"prior_cache must be one of {PRIOR_CACHES}: {prior_cache}")
    by_hand = prior_cache == "hand"
    if stats is not None:
        stats.searches += 1
    rng = rng or random.Random()
    me = obs.player
    classes = _equivalence_classes(obs, use_representatives)
    root_moves = list(classes)
    root = Node(None, pybl.to_opponent(me), None)

    # ルートの事前確率 (代表の手には, 同等な手の確率の合計を割り当てる)
    planes, scalars = F.encode(obs)
    logits, values = yield planes[None], scalars[None]
    legal = obs.get_legal_moves()
    probs = dict(zip(legal.tolist(), _softmax(logits[0][legal])))
    root_prior = np.array([sum(probs[m] for m in classes[r]) for r in root_moves])
    if dirichlet_alpha > 0 and len(root_moves) > 1:
        noise = np.random.default_rng(rng.getrandbits(64)).dirichlet([dirichlet_alpha] * len(root_moves))
        root_prior = (1 - dirichlet_epsilon) * root_prior + dirichlet_epsilon * noise
    root_priors = np.ascontiguousarray(root_prior, dtype=np.float64)
    root_value = float(values[0])

    if sample_world is None:
        base = obs.sample_state(seed=rng.getrandbits(64))

        def sample_world(rng):
            work = base.copy()
            work.determinize(me, seed=rng.getrandbits(64))
            return work
    root_masks = F.flag_card_masks(obs)      # フラッグに置かれたカード (determinization で変わらない)
    done = 0
    while done < num_simulations:
        pending = []        # (葉のノード, 経路, 葉の局面の手番のプレイヤー, その手札 (by_hand か stats のときだけ))
        inputs = []
        for _ in range(min(batch_size, num_simulations - done)):
            done += 1
            work = sample_world(rng)
            node, path = root, []
            masks = list(root_masks)
            while True:
                if work.is_terminal:
                    _backup(path, lambda p, s=work: _terminal_value(s, p), virtual=True)
                    if stats is not None:
                        stats.record_leaf("terminal", len(path))
                    break
                logits = None       # None なら _select は node.logits を使う
                if node is not root:
                    if by_hand:
                        hand = int(work.get_hand(work.side_to_move))
                        logits = node.cache.get(hand) if node.cache is not None else None
                        if stats is not None:
                            stats.record_arrive(len(path), work.side_to_move == me, logits is None)
                        if logits is None:
                            # この手札ではまだ評価していないので, ここを葉として今の局面を評価する
                            planes, scalars = F.encode(work.observe(work.side_to_move), masks)
                            inputs.append((planes, scalars))
                            pending.append((node, path, work.side_to_move, hand))
                            break
                    elif node.logits is None:
                        if node.visits > 1:     # 同じバッチの別のシミュレーションが評価待ちにしたノード
                            _backup(path, lambda p: 0.0, virtual=True)
                            if stats is not None:
                                stats.record_leaf("collision", len(path))
                            break
                        o = work.observe(work.side_to_move)
                        planes, scalars = F.encode(o, masks)
                        inputs.append((planes, scalars))
                        hand = int(work.get_hand(work.side_to_move)) if stats is not None else None
                        pending.append((node, path, work.side_to_move, hand))
                        break
                    if stats is not None:
                        stats.record_select(len(path), work.side_to_move == me,
                                            hand if by_hand else node.hand, work, node)
                move = _select(node, root, root_moves, root_priors, work, c_puct, fpu, fpu_reduction, logits)
                mover = work.side_to_move
                child = node.children.get(move)
                if child is None:
                    child = Node(move, mover, node)
                    node.children[move] = child
                work.play(move)
                if move != _PASS:
                    masks[int(mover) * F.NUM_FLAGS + move % F.NUM_FLAGS] |= 1 << (move // F.NUM_FLAGS)
                node = child
                path.append(node)
                node.visits += 1            # 仮想損失: 評価を待つ間は負けとして数える
                node.value_sum -= 1.0

        if pending:
            logits, values = yield np.stack([x[0] for x in inputs]), np.stack([x[1] for x in inputs])
            for (node, path, to_move, hand), node_logits, value in zip(pending, logits, values):
                if by_hand:
                    if node.cache is None:
                        node.cache = {}
                    if hand not in node.cache:      # 同じバッチで同じ手札を2回評価したときは先のものを使う
                        node.cache[hand] = np.ascontiguousarray(node_logits, dtype=np.float32)
                elif node.logits is None:
                    node.logits = np.ascontiguousarray(node_logits, dtype=np.float32)
                    node.hand = hand
                if stats is not None:
                    stats.record_leaf("eval", len(path))
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


def revealed_world_sampler(state, reveal: str = "hand"):
    """本当の局面 state から, シミュレーションごとの局面を作る関数を返す (診断用. 相手の情報を見る「ずる」をする).

    reveal="hand": 相手の手札は本当のものを使い, 山札の順番だけをシミュレーションごとにランダムにする
    reveal="all":  山札の順番も含めて, 本当の局面をそのまま使う
    """
    if reveal == "all":
        return lambda rng: state.copy()
    if reveal != "hand":
        raise ValueError(f"reveal must be 'hand' or 'all': {reveal}")
    players = (pybl.FIRST, pybl.SECOND)
    cards = lambda bits: [int(c) for c in pybl.CardIterator(bits)]
    hands = tuple(cards(state.get_hand(p)) for p in players)
    flags = [tuple([int(c) for c in state.get_flag_cards(i, p)] for p in players) for i in range(F.NUM_FLAGS)]
    owners = [None if o == pybl.NULL_PLAYER else o for o in (state.get_flag_owner(i) for i in range(F.NUM_FLAGS))]
    completers = [None if c == pybl.NULL_PLAYER else c
                  for c in (state.get_flag_first_completer(i) for i in range(F.NUM_FLAGS))]
    used = int(state.board_cards) | int(state.get_hand(players[0])) | int(state.get_hand(players[1]))
    deck = cards(int(pybl.ALL_CARDS) & ~used)
    side, first = state.side_to_move, state.first_player

    def sample(rng):
        order = list(deck)
        rng.shuffle(order)
        world = pybl.GameState()
        world.set_position(side, hands, flags, order, owners, completers, first)
        return world
    return sample


def _select_py(node, root, root_moves: list, root_priors: np.ndarray, work, c_puct: float, fpu: float,
               fpu_reduction: float | None = None, node_logits: np.ndarray | None = None) -> int:
    """PUCT で手を選ぶ (root_priors は root_moves の順のルートの事前確率). Cython 版 (_fast.select) と同じ計算をする.
    node_logits: ルート以外の事前確率に使うロジット (省略時は node.logits)"""
    if node is root:
        legal = root_moves
        priors = root_priors
    else:
        legal = work.get_legal_moves()
        priors = _softmax((node.logits if node_logits is None else node_logits)[legal].astype(np.float64))
        legal = legal.tolist()
    children = node.children
    total, value_total, visited_prior = 0, 0.0, 0.0
    for m, p in zip(legal, priors):
        child = children.get(m)
        if child is not None and child.visits > 0:
            total += child.visits
            value_total += child.value_sum
            visited_prior += p
    sqrt_total = math.sqrt(max(total, 1))
    if fpu_reduction is not None:
        # 子の価値はその手を指したプレイヤー (このノードの手番) から見た値なので, その平均を親の価値とする
        fpu = (value_total / total if total > 0 else 0.0) - fpu_reduction * math.sqrt(visited_prior)
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


try:
    from az._fast import select as _select
except ImportError:
    _select = _select_py
