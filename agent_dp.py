"""Deep Barca (McCulloch et al., MAICS 2017) の戦略に基づくエージェント (部隊カードのみ).

探索は行わず, 合法手ごとに「その手を指した直後の局面でゲームに勝つ確率」を確率モデルで見積もり,
最大の手を選ぶ (1手先の評価). 勝つ確率は次の3段階で求める.

1. 各フラッグで, 両プレイヤーが作りうる上位4つのフォーメーションを求める
   - 自分: 自分の手札と見えないカード (相手の手札 + 山札) から作れるもの
   - 相手: 見えないカードから作れるもの (自分の手札のカードは使えない)
2. 各フラッグを確保する確率を求める
   - 確保済み: 1 または 0 / 両側とも空: 0.5
   - 両側にカードがある: MFA (自分の上位4つの役のいずれかで勝つ確率)
   - 片側だけにカードがある: MFA と BSFA (カードの無い側の最強の1つの役との比較) の混合
3. 「5本確保」と「3本連続」で勝つ確率を, 山札の残り枚数に応じた重みで平均する
   (序盤は5本確保を, 終盤は3本連続を重視する)

カードの確率 (論文 3.4):
- そのフラッグの同じ側に置かれている: 1 / 盤面の他の場所にある: 0
- 自分の手札にある: 自分の役なら 1, 相手の役なら 0
- 見えない: 山札は両者に半分ずつ配られるとみなし, 自分が引く確率 p = (山札の枚数 / 2) / 見えないカードの枚数,
  相手が引く (または既に持っている) 確率 1 - p

負けが確定したフラッグへの捨て札 (論文 3.5):
相手が次の手番で証明により確保できるフラッグに自分のカードを置ける場合は, そこに捨て札をする.
捨てるカードは, 相手の上位の役が必要とし (wishlist に多く現れ), 自分の上位の役が必要としないものを選ぶ.

論文で明記されていない点と, PyBL のルールに合わせた点:
- 「役」は強さ (種類と合計) が同じカードの組のまとまりとし (論文の表1の「9-9-9」に合わせる),
  その完成確率は, いずれかの組の全カードが手に入る確率とする (各カードは独立. 枚数が多い場合は近似)
- 「上回られる確率」「いずれかの役で勝つ確率」(和集合) は, 各役を独立とみなして 1 - Π(1 - p) で求める.
  ただし自分の役が必ず使うカードを含む相手の組は, 自分の役が完成した世界では作れないため除く
- 同じ強さの役どうしは, 先に完成させた側が勝つため, 勝敗は五分 (0.5) とみなす
- MFA と BSFA の混合比は MIX_WEIGHT, 5本確保と3本連続の重みは山札の残り枚数 / 46 とする
- 相手の wishlist は, 盤面に出ていない全てのカード (自分の手札を含む) で作れる上位の役から作る
  (盤面に出したカードは証明に使えるため, 相手の役に必要なカードを捨てると証明を助けられる)
- PyBL のルールでは確保の判定が着手の直後に行われる. そのため捨て札は, 相手が次の手番で証明できる
  (自分の手番の時点で既に証明が成り立っている) フラッグに限って行う

使い方:
    import pybl
    from agent_dp import deep_barca_move

    state = pybl.GameState()
    move = deep_barca_move(state.observe(state.side_to_move))
    state.play(move)

デバッグ表示:
    deep_barca_move(obs, debug=True) とするか, このファイルの DEBUG を True にすると,
    現在の局面の評価, 上位の候補手, 選んだ手の後の各フラッグの確保確率と両者の上位の役などを
    標準エラー出力に表示する (match.py の出力とは混ざらない).
"""
import itertools
import random
import sys

import pybl

TOP_FORMATIONS = 4          # 各フラッグで考える上位の役の数 (論文 3.2)
MIX_WEIGHT = 0.5            # 片側だけにカードがあるフラッグでの MFA の重み (残りは BSFA)
MAX_EXACT_CARDS = 10        # 役の完成確率を厳密に計算する, 不確かなカードの枚数の上限

DEBUG = False               # True にすると, 着手決定の過程 (各フラッグの確保確率など) を標準エラー出力に表示する
DEBUG_TOP_MOVES = 5         # デバッグ表示で評価値を表示する候補手の数
INITIAL_DECK = pybl.NUM_CARDS - 2 * pybl.HAND_SIZE   # 配り終えた時点の山札の枚数 (46)

_default_rng = random.Random()

_COLOR = [int(pybl.card_color(c)) for c in range(pybl.NUM_CARDS)]
_RANK = [pybl.card_rank(c) for c in range(pybl.NUM_CARDS)]
_CARD = [[int(pybl.make_card(color, rank)) if rank >= 1 else -1 for rank in range(pybl.NUM_RANKS + 1)]
         for color in range(pybl.NUM_COLORS)]


def _bit(card: int) -> int:
    return 1 << card


# ---------------------------------------------------------------------------
# フォーメーションの列挙 (論文 3.2 の「上位4つの役」)
# ---------------------------------------------------------------------------

def _wedges(placed: list, pool: int):
    colors = {_COLOR[c] for c in placed}
    if len(colors) > 1:
        return
    for color in (colors if colors else range(pybl.NUM_COLORS)):
        for low in range(1, pybl.NUM_RANKS - 1):
            cards = [_CARD[color][low + k] for k in range(3)]
            if all(c in cards for c in placed) and all(c in placed or pool & _bit(c) for c in cards):
                yield cards


def _phalanxes(placed: list, pool: int, need: int):
    ranks = {_RANK[c] for c in placed}
    if len(ranks) > 1:
        return
    for rank in (ranks if ranks else range(pybl.NUM_RANKS, 0, -1)):
        available = [_CARD[color][rank] for color in range(pybl.NUM_COLORS) if pool & _bit(_CARD[color][rank])]
        for extra in itertools.combinations(available, need):
            yield placed + list(extra)


def _battalions(placed: list, pool: int, need: int):
    colors = {_COLOR[c] for c in placed}
    if len(colors) > 1:
        return
    for color in (colors if colors else range(pybl.NUM_COLORS)):
        available = [_CARD[color][r] for r in range(pybl.NUM_RANKS, 0, -1) if pool & _bit(_CARD[color][r])]
        for extra in itertools.combinations(available, need):
            yield placed + list(extra)


def _skirmishers(placed: list, pool: int):
    ranks = [_RANK[c] for c in placed]
    if len(set(ranks)) != len(ranks):
        return
    for low in range(pybl.NUM_RANKS - 2, 0, -1):
        window = range(low, low + 3)
        if any(r not in window for r in ranks):
            continue
        choices = [[_CARD[color][r] for color in range(pybl.NUM_COLORS) if pool & _bit(_CARD[color][r])]
                   for r in window if r not in ranks]
        for extra in itertools.product(*choices):
            yield placed + list(extra)


def _hosts(placed: list, pool: int, need: int):
    # 合計が大きい役だけが上位に入るため, 数字の大きいカードに限って組み合わせる
    top = sorted((c for c in range(pybl.NUM_CARDS) if pool & _bit(c)), key=lambda c: -_RANK[c])[:10]
    for extra in itertools.combinations(top, need):
        yield placed + list(extra)


def _cards_of(mask: int) -> list:
    return [c for c in range(pybl.NUM_CARDS) if mask >> c & 1]


def _any_set_probability(sets: list, card_prob: dict) -> float:
    """sets (カードの組のビットマスク) のいずれかの組の全てのカードが手に入る確率.

    各カードが手に入るかは独立とし, 確率が 0 と 1 の間のカードが MAX_EXACT_CARDS 枚以下なら厳密に,
    それより多ければ各組を独立とみなして近似する.
    """
    if not sets:
        return 0.0
    uncertain = sorted({c for s in sets for c in _cards_of(s) if card_prob[c] < 1.0})
    if len(uncertain) > MAX_EXACT_CARDS:
        q = 1.0
        for s in sets:
            p = 1.0
            for c in _cards_of(s):
                p *= card_prob[c]
            q *= 1.0 - p
        return 1.0 - q

    index = {c: i for i, c in enumerate(uncertain)}
    needs = [sum(1 << index[c] for c in _cards_of(s) if c in index) for s in sets]
    probs = [card_prob[c] for c in uncertain]
    total = 0.0
    for owned in range(1 << len(uncertain)):
        if any(need & ~owned == 0 for need in needs):
            p = 1.0
            for i, q in enumerate(probs):
                p *= q if owned >> i & 1 else 1.0 - q
            total += p
    return total


class Level:
    """強さ (フォーメーションの種類と合計) が同じ役のまとまり. 論文の「役」に当たる.

    例えば「9 のスリーカード」は, 使う9の色の組み合わせが違っても1つの役とし, その完成確率は
    いずれかの組み合わせが完成する確率とする (論文の表1で 0.82).
    """
    __slots__ = ("strength", "sets", "card_prob", "prob", "core")

    def __init__(self, strength: int, sets: list, card_prob: dict):
        self.strength = strength
        self.sets = sets                # この強さになるカードの組 (ビットマスク) のリスト
        self.card_prob = card_prob      # カード -> そのプレイヤーが手に入れる確率
        self.prob = _any_set_probability(sets, card_prob)
        core = sets[0]
        for mask in sets[1:]:
            core &= mask
        self.core = core                # この役が完成すれば必ず使われるカード

    def cards(self) -> int:
        """この役に関わるカード全体 (ビットマスク)"""
        mask = 0
        for s in self.sets:
            mask |= s
        return mask


def _top_formations(placed: tuple, pool: int, prob, k: int = TOP_FORMATIONS) -> list:
    """placed (そのフラッグに置いたカード) に pool のカードを加えて作れる役のうち, 強い順に上位 k 個 (Level) を返す.

    prob はカード -> そのプレイヤーが手に入れる確率. 完成確率が0の組は除く.
    """
    placed = list(placed)
    placed_mask = sum(_bit(c) for c in placed)
    if len(placed) == pybl.FORMATION_SIZE:
        return [Level(pybl.evaluate_formation(*placed), [placed_mask], {c: 1.0 for c in placed})]

    need = pybl.FORMATION_SIZE - len(placed)
    pool &= ~placed_mask
    generators = (_wedges(placed, pool), _phalanxes(placed, pool, need), _battalions(placed, pool, need),
                  _skirmishers(placed, pool), _hosts(placed, pool, need))

    card_prob = {c: 1.0 for c in placed}
    levels = {}         # 強さ -> カードの組 (ビットマスク) の集合
    for generator in generators:
        for cards in generator:
            for c in cards:
                if c not in card_prob:
                    card_prob[c] = prob(c)
            if any(card_prob[c] == 0.0 for c in cards):
                continue
            levels.setdefault(pybl.evaluate_formation(*cards), set()).add(sum(_bit(c) for c in cards))
        # 強い種類から順に列挙しているので, k 個以上の強さが見つかれば以降の種類は上位に入らない
        if len(levels) >= k:
            break

    top = sorted(levels, reverse=True)[:k]
    return [Level(strength, sorted(levels[strength]), card_prob) for strength in top]


# ---------------------------------------------------------------------------
# フラッグを確保する確率 (論文 3.3)
# ---------------------------------------------------------------------------

def _union(probabilities) -> float:
    """独立な事象のいずれかが起きる確率"""
    q = 1.0
    for p in probabilities:
        q *= 1.0 - p
    return 1.0 - q


def _beaten_probability(formation: Level, opponent_formations: list) -> float:
    """formation が完成したとき, 相手の役のいずれかに上回られる確率 (同じ強さは0.5).

    formation が必ず使うカードを含む相手の組は, formation が完成した世界では作れないため除く.
    """
    probabilities = []
    for g in opponent_formations:
        if g.strength < formation.strength:
            continue
        sets = [s for s in g.sets if not s & formation.core]
        p = g.prob if len(sets) == len(g.sets) else _any_set_probability(sets, g.card_prob)
        probabilities.append(p if g.strength > formation.strength else 0.5 * p)
    return _union(probabilities)


def _mfa(mine: list, theirs: list) -> float:
    """自分の役のいずれかで勝つ確率. 各役で勝つ確率 = 完成する確率 × 上回られない確率"""
    return _union(f.prob * (1.0 - _beaten_probability(f, theirs)) for f in mine)


def _flag_probability(mine: list, theirs: list, my_count: int, their_count: int) -> float:
    """自分がフラッグを確保する確率. mine / theirs は両者の上位の役, *_count は置いたカードの枚数"""
    if my_count == 0 and their_count == 0:
        return 0.5
    if not mine and not theirs:
        return 0.5          # どちらも役を完成できない
    mfa = _mfa(mine, theirs)
    if my_count > 0 and their_count > 0:
        return mfa
    if their_count == 0:
        # 相手は自分の役を見て, それを上回る最強の1つの役を狙う
        bsfa = _mfa(mine, theirs[:1])
    else:
        # 自分は相手の役を上回る最強の1つの役を狙う
        bsfa = _mfa(mine[:1], theirs)
    return MIX_WEIGHT * mfa + (1.0 - MIX_WEIGHT) * bsfa


# ---------------------------------------------------------------------------
# ゲームに勝つ確率 (論文 3.2)
# ---------------------------------------------------------------------------

def _majority_probability(flag_probs: list) -> float:
    """5本以上確保する確率"""
    dist = [1.0]                    # dist[n] = n 本確保する確率
    for p in flag_probs:
        new = [0.0] * (len(dist) + 1)
        for n, q in enumerate(dist):
            new[n] += q * (1.0 - p)
            new[n + 1] += q * p
        dist = new
    return sum(dist[5:])


def _adjacent_probability(flag_probs: list) -> float:
    """3本連続で確保する確率"""
    run = [1.0, 0.0, 0.0]           # run[k] = 末尾の連続確保数が k で, まだ3本連続が無い確率
    for p in flag_probs:
        run = [(run[0] + run[1] + run[2]) * (1.0 - p), run[0] * p, run[1] * p]
    return 1.0 - sum(run)


def _game_probability(flag_probs: list, deck_count: int) -> float:
    w = min(1.0, deck_count / INITIAL_DECK)
    return w * _majority_probability(flag_probs) + (1.0 - w) * _adjacent_probability(flag_probs)


# ---------------------------------------------------------------------------
# エージェント
# ---------------------------------------------------------------------------

class _Evaluator:
    """1回の着手決定の中で, フォーメーションの列挙結果を使い回しながら局面を評価する"""

    def __init__(self, obs: pybl.Observation):
        self.me = obs.player
        self.opp = pybl.to_opponent(self.me)
        self.unseen = int(obs.unseen_cards)
        self.deck_count = obs.deck_count
        unseen_count = bin(self.unseen).count("1")
        # 山札は両者に半分ずつ配られるとみなす (論文 3.4)
        self.p_me = (self.deck_count / 2) / unseen_count if unseen_count else 0.0
        self.cache = {}

    def formations(self, placed: tuple, player, hand: int) -> list:
        key = (placed, player == self.me, hand if player == self.me else 0)
        if key not in self.cache:
            if player == self.me:
                pool = hand | self.unseen
                prob = lambda c: 1.0 if hand & _bit(c) else self.p_me
            else:
                pool = self.unseen
                prob = lambda c: 1.0 - self.p_me
            self.cache[key] = _top_formations(placed, pool, prob)
        return self.cache[key]

    def evaluate(self, flags: list, owners: list, hand: int) -> float:
        """flags[i] = (自分のカード, 相手のカード), owners[i] = 確保者 から, ゲームに勝つ確率を求める"""
        flag_probs = []
        for (mine, theirs), owner in zip(flags, owners):
            if owner == self.me:
                flag_probs.append(1.0)
            elif owner == self.opp:
                flag_probs.append(0.0)
            else:
                flag_probs.append(_flag_probability(self.formations(mine, self.me, hand),
                                                    self.formations(theirs, self.opp, hand),
                                                    len(mine), len(theirs)))
        return _game_probability(flag_probs, self.deck_count)

    def details(self, flags: list, owners: list, hand: int) -> dict:
        """デバッグ表示用に, ゲームに勝つ確率の内訳と各フラッグの計算の詳細を返す"""
        rows = []
        for (mine, theirs), owner in zip(flags, owners):
            row = {"mine": mine, "theirs": theirs, "my_forms": [], "their_forms": [], "mfa": None, "bsfa": None}
            if owner == self.me:
                row.update(prob=1.0, method="確保済(自分)")
            elif owner == self.opp:
                row.update(prob=0.0, method="確保済(相手)")
            else:
                my_forms = self.formations(mine, self.me, hand)
                their_forms = self.formations(theirs, self.opp, hand)
                row.update(my_forms=my_forms, their_forms=their_forms,
                           prob=_flag_probability(my_forms, their_forms, len(mine), len(theirs)))
                if not mine and not theirs:
                    row["method"] = "両側とも空"
                elif mine and theirs:
                    row.update(method="MFA", mfa=_mfa(my_forms, their_forms))
                else:
                    bsfa = _mfa(my_forms, their_forms[:1]) if mine else _mfa(my_forms[:1], their_forms)
                    row.update(method="MFA+BSFA", mfa=_mfa(my_forms, their_forms), bsfa=bsfa)
            rows.append(row)
        probs = [row["prob"] for row in rows]
        weight = min(1.0, self.deck_count / INITIAL_DECK)
        return {"flags": rows, "majority": _majority_probability(probs), "adjacent": _adjacent_probability(probs),
                "weight": weight, "total": _game_probability(probs, self.deck_count)}


# ---------------------------------------------------------------------------
# デバッグ表示
# ---------------------------------------------------------------------------

_FORMATION_NAMES = {pybl.WEDGE: "ウェッジ", pybl.PHALANX: "ファランクス", pybl.BATTALION: "バタリオン",
                    pybl.SKIRMISHER: "スカーミッシャー", pybl.HOST: "ホスト"}


def _debug(text: str = "") -> None:
    print(f"[dp] {text}", file=sys.stderr)


def _cards_str(cards) -> str:
    return " ".join(pybl.card_to_str(c) for c in cards) or "-"


def _formations_str(levels: list) -> str:
    return ", ".join(f"{_FORMATION_NAMES[int(pybl.strength_to_formation_type(f.strength))]}"
                     f"{pybl.strength_to_sum(f.strength)}({f.prob:.2f})" for f in levels) or "なし"


def _summary_str(details: dict) -> str:
    return (f"勝つ確率 {details['total']:.3f} (5本確保 {details['majority']:.3f}, 3本連続 {details['adjacent']:.3f}, "
            f"5本確保の重み {details['weight']:.2f})")


def _print_flags(details: dict) -> None:
    _debug("No  自分       相手       確保確率  計算方法")
    for i, row in enumerate(details["flags"]):
        method = row["method"]
        if row["mfa"] is not None and row["bsfa"] is not None:
            method += f" (MFA {row['mfa']:.3f}, BSFA {row['bsfa']:.3f})"
        _debug(f"{i + 1:>2}  {_cards_str(row['mine']):<10} {_cards_str(row['theirs']):<10} "
               f"{row['prob']:>6.3f}    {method}")
        if row["my_forms"] or row["their_forms"]:
            _debug(f"      自分の上位の役: {_formations_str(row['my_forms'])}")
            _debug(f"      相手の上位の役: {_formations_str(row['their_forms'])}")


def _flags_of(state, me) -> tuple:
    opp = pybl.to_opponent(me)
    flags = [(tuple(sorted(int(c) for c in state.get_flag_cards(i, me))),
              tuple(sorted(int(c) for c in state.get_flag_cards(i, opp)))) for i in range(pybl.NUM_FLAGS)]
    owners = [int(state.get_flag_owner(i)) for i in range(pybl.NUM_FLAGS)]
    return flags, owners


def _discard_move(obs: pybl.Observation, flags: list, owners: list, evaluator: _Evaluator):
    """相手が次の手番で証明により確保できるフラッグへの捨て札 (論文 3.5).

    (着手, デバッグ用の情報) を返す. 捨て札をしない場合は (None, None)
    """
    me, opp = obs.player, pybl.to_opponent(obs.player)
    board = int(obs.board_cards)
    hand = int(obs.hand)
    doomed = []
    for i, ((mine, theirs), owner) in enumerate(zip(flags, owners)):
        if owner != pybl.NULL_PLAYER or len(mine) == pybl.FORMATION_SIZE:
            continue
        first, second = (mine, theirs) if me == pybl.FIRST else (theirs, mine)
        completer = obs.get_flag_first_completer(i)
        completer = None if completer == pybl.NULL_PLAYER else completer
        if pybl.judge_flag(first, second, board, opp, completer) == opp:
            doomed.append(i)
    if not doomed:
        return None, None

    # wishlist: 各フラッグで両者の上位の役が必要とする (まだ置かれていない) カードの出現回数
    off_board = hand | evaluator.unseen
    my_wish, their_wish = {}, {}
    for (mine, theirs), owner in zip(flags, owners):
        if owner != pybl.NULL_PLAYER:
            continue
        for level in evaluator.formations(mine, me, hand):
            for c in _cards_of(level.cards()):
                my_wish[c] = my_wish.get(c, 0) + 1
        for level in _top_formations(theirs, off_board, lambda c: 1.0):
            for c in _cards_of(level.cards()):
                their_wish[c] = their_wish.get(c, 0) + 1

    hand_cards = [int(c) for c in pybl.CardIterator(hand)]
    card = max(hand_cards, key=lambda c: (their_wish.get(c, 0) - my_wish.get(c, 0), -_RANK[c]))
    info = {"doomed": doomed,
            "wish": {c: (their_wish.get(c, 0), my_wish.get(c, 0)) for c in hand_cards}}
    return pybl.make_move(card, doomed[0]), info


def deep_barca_move(obs: pybl.Observation, rng: random.Random | None = None, debug: bool | None = None):
    """obs (手番のプレイヤーから見える情報) をもとに, Deep Barca の確率モデルで着手を選んで返す.

    obs には手番のプレイヤーの pybl.Observation (GameState.observe で取得) を渡す.
    rng: random.Random を渡すと, 評価値が同じ手の中からの選択にその乱数を使う (再現性が必要な場合に用いる)
    debug: True なら着手決定の過程を標準エラー出力に表示する (省略時はこのファイルの DEBUG に従う)

    置ける場所が無い場合は pybl.PASS_MOVE を返す. 終局している場合は ValueError を送出する.
    """
    if not isinstance(obs, pybl.Observation):
        raise TypeError("obs must be a pybl.Observation (use GameState.observe())")
    rng = rng or _default_rng
    debug = DEBUG if debug is None else debug
    moves = obs.get_legal_moves().tolist()
    if len(moves) == 0:
        if obs.is_terminal:
            raise ValueError("the game is already over")
        raise ValueError("it is not the observing player's turn")

    me = obs.player
    if debug:
        _debug(f"===== {pybl.player_to_str(me)} の手番 / 山札 {obs.deck_count} 枚 / "
               f"見えないカード {bin(int(obs.unseen_cards)).count('1')} 枚 / 手札 {_cards_str(pybl.CardIterator(obs.hand))} =====")
    if len(moves) == 1:
        if debug:
            _debug(f"合法手は {pybl.move_to_str(moves[0])} のみ")
        return pybl.Move(moves[0])

    win = pybl.FIRST_WIN if me == pybl.FIRST else pybl.SECOND_WIN
    evaluator = _Evaluator(obs)
    hand = int(obs.hand)
    if debug:
        flags, owners = _flags_of(obs, me)
        _debug(f"自分が見えないカードを引く確率 p = {evaluator.p_me:.3f}")
        _debug(f"現在の局面: {_summary_str(evaluator.details(flags, owners, hand))}")

    # 着手直後の盤面 (フラッグの確保を含む) を得るための局面. 盤面と確保の判定は見えないカードの配分に依存しない
    sample = obs.sample_state(seed=rng.getrandbits(64))

    scores = {}
    after = {}          # デバッグ表示用: 着手 -> (着手後のフラッグ, 確保者, 手札)
    for move in moves:
        sample.play(move)
        if sample.is_terminal:
            score = 2.0 if sample.result == win else 0.5 if sample.result == pybl.DRAW else -1.0
        else:
            flags, owners = _flags_of(sample, me)
            # 補充で引くカードは考慮しない (論文と同じく, カードを置いた直後の局面を評価する)
            # move_card は numpy.int8 を返すため, Python の int に変換してからビット演算する
            # (numpy のまま計算すると, Windows では 32 ビットの整数として扱われてオーバーフローする)
            hand_after = hand & ~_bit(int(pybl.move_card(move)))
            score = evaluator.evaluate(flags, owners, hand_after)
            if debug:
                after[move] = (flags, owners, hand_after)
        sample.undo()
        scores[move] = score
        if score >= 2.0:
            if debug:
                _debug(f"選んだ手: {pybl.move_to_str(move)} (勝ちが決まる手)")
            return pybl.Move(move)      # 勝ちが決まる手

    if debug:
        ranked = sorted(moves, key=lambda m: scores[m], reverse=True)
        _debug(f"候補手 (全 {len(moves)} 手中, 評価値の上位 {min(DEBUG_TOP_MOVES, len(moves))} 手):")
        for rank, move in enumerate(ranked[:DEBUG_TOP_MOVES], 1):
            if move in after:
                _debug(f"  {rank}. {pybl.move_to_str(move):<7} {_summary_str(evaluator.details(*after[move]))}")
            else:
                _debug(f"  {rank}. {pybl.move_to_str(move):<7} 評価値 {scores[move]:.3f} (終局)")

    flags, owners = _flags_of(obs, me)
    discard, info = _discard_move(obs, flags, owners, evaluator)
    if discard is not None:
        if debug:
            _debug(f"選んだ手: {pybl.move_to_str(discard)} (捨て札: フラッグ "
                   f"{', '.join(str(i + 1) for i in info['doomed'])} は相手が次の手番で証明により確保できる)")
            _debug("  手札の各カードの wishlist での出現回数 (相手 / 自分): " + ", ".join(
                f"{pybl.card_to_str(c)} {t}/{m}" for c, (t, m) in sorted(info["wish"].items())))
            if discard in after:
                _print_flags(evaluator.details(*after[discard]))
        return pybl.Move(discard)

    best = max(scores.values())
    candidates = [m for m in moves if scores[m] >= best - 1e-12]
    move = candidates[rng.randrange(len(candidates))]
    if debug:
        tie = f", 同点 {len(candidates)} 手からランダム" if len(candidates) > 1 else ""
        _debug(f"選んだ手: {pybl.move_to_str(move)} (評価値が最大{tie})")
        _debug("選んだ手の後の各フラッグ:")
        _print_flags(evaluator.details(*after[move]))
    return pybl.Move(move)
