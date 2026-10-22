"""ネットワークの入出力のエンコードと, 色・フラッグの対称変換.

入力 (手番のプレイヤーから見える情報 pybl.Observation のみを使う):
    planes:  (NUM_PLANES, 6, 10) の 0/1 の配列. 軸は (チャンネル, 色, 数字 - 1)
        0       自分の手札
        1       見えないカード (相手の手札 + 山札)
        2〜10   フラッグ 1〜9 の自分側に置かれたカード
        11〜19  フラッグ 1〜9 の相手側に置かれたカード
    scalars: (NUM_SCALARS,) の配列
        0〜8    フラッグ 1〜9 を自分が確保済み
        9〜17   フラッグ 1〜9 を相手が確保済み
        18〜26  フラッグ 1〜9 で自分が先に役を完成させた
        27〜35  フラッグ 1〜9 で相手が先に役を完成させた
        36      山札の枚数 / 46
        37      相手の手札の枚数 / 7
        38      自分の手札の枚数 / 7
        39      直前の手がパスだったか
        40      自分が先手か

出力 (方策) の行動の番号は PyBL の着手の値をそのまま使う:
    カード * 9 + フラッグ (0〜539), パス = 540. 行動の数は NUM_ACTIONS (541)

対称変換 (Symmetry):
    バトルラインのルールは, 色の入れ替え (6! = 720 通り) とフラッグの左右反転 (2 通り) で変わらない.
    この 1440 通りの変換で, 入力・方策・合法手のマスクを同じように並べ替えられる (価値は変わらない).
    学習データの拡張に使う.
"""
import itertools

import numpy as np

import pybl

NUM_COLORS = pybl.NUM_COLORS
NUM_RANKS = pybl.NUM_RANKS
NUM_CARDS = pybl.NUM_CARDS
NUM_FLAGS = pybl.NUM_FLAGS

NUM_ACTIONS = int(pybl.PASS_MOVE) + 1           # 541
PASS_ACTION = int(pybl.PASS_MOVE)               # 540

PLANE_HAND = 0
PLANE_UNSEEN = 1
PLANE_MY_FLAG = 2                               # 2〜10
PLANE_OPP_FLAG = PLANE_MY_FLAG + NUM_FLAGS      # 11〜19
NUM_PLANES = PLANE_OPP_FLAG + NUM_FLAGS         # 20
PLANE_SHAPE = (NUM_COLORS, NUM_RANKS)

SCALAR_MY_CLAIMED = 0
SCALAR_OPP_CLAIMED = SCALAR_MY_CLAIMED + NUM_FLAGS
SCALAR_MY_FIRST = SCALAR_OPP_CLAIMED + NUM_FLAGS
SCALAR_OPP_FIRST = SCALAR_MY_FIRST + NUM_FLAGS
SCALAR_DECK = SCALAR_OPP_FIRST + NUM_FLAGS      # 36
SCALAR_OPP_HAND = SCALAR_DECK + 1
SCALAR_MY_HAND = SCALAR_OPP_HAND + 1
SCALAR_PASSED = SCALAR_MY_HAND + 1
SCALAR_FIRST_PLAYER = SCALAR_PASSED + 1
NUM_SCALARS = SCALAR_FIRST_PLAYER + 1           # 41

_INITIAL_DECK = NUM_CARDS - 2 * pybl.HAND_SIZE

# カードの番号は 色 * 10 + (数字 - 1) なので, planes を (チャンネル, 60) に変形すると番号で直接書き込める
assert all(int(pybl.make_card(c, r)) == c * NUM_RANKS + r - 1
           for c in range(NUM_COLORS) for r in range(1, NUM_RANKS + 1))


def _cards(bits) -> list:
    return [int(c) for c in pybl.CardIterator(bits)]


def flag_card_masks(obs) -> list:
    """各フラッグに置かれたカードのビットマスクのリスト. 要素 [プレイヤー × 9 + フラッグ] (プレイヤーは先手 0, 後手 1).

    フラッグのカードは determinization で変わらないので, 探索ではルートで1回求めて, 手を指すたびに更新して encode に渡す.
    """
    masks = [0] * (2 * NUM_FLAGS)
    for player in (pybl.FIRST, pybl.SECOND):
        for i in range(NUM_FLAGS):
            for c in obs.get_flag_cards(i, player):
                masks[int(player) * NUM_FLAGS + i] |= 1 << int(c)
    return masks


def _encode_py(obs: pybl.Observation, flag_masks: list | None = None) -> tuple:
    """encode の Python 版"""
    me = obs.player
    opp = pybl.to_opponent(me)
    planes = np.zeros((NUM_PLANES, NUM_CARDS), dtype=np.float32)
    planes[PLANE_HAND, _cards(obs.hand)] = 1.0
    planes[PLANE_UNSEEN, _cards(obs.unseen_cards)] = 1.0

    if flag_masks is None:
        flag_masks = flag_card_masks(obs)
    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    for i in range(NUM_FLAGS):
        planes[PLANE_MY_FLAG + i, _cards(flag_masks[int(me) * NUM_FLAGS + i])] = 1.0
        planes[PLANE_OPP_FLAG + i, _cards(flag_masks[int(opp) * NUM_FLAGS + i])] = 1.0
        owner = obs.get_flag_owner(i)
        completer = obs.get_flag_first_completer(i)
        scalars[SCALAR_MY_CLAIMED + i] = owner == me
        scalars[SCALAR_OPP_CLAIMED + i] = owner == opp
        scalars[SCALAR_MY_FIRST + i] = completer == me
        scalars[SCALAR_OPP_FIRST + i] = completer == opp

    scalars[SCALAR_DECK] = obs.deck_count / _INITIAL_DECK
    scalars[SCALAR_OPP_HAND] = obs.opponent_hand_count / pybl.HAND_SIZE
    scalars[SCALAR_MY_HAND] = obs.get_hand_count(me) / pybl.HAND_SIZE
    scalars[SCALAR_PASSED] = obs.consecutive_pass_count > 0
    scalars[SCALAR_FIRST_PLAYER] = me == obs.first_player
    return planes.reshape(NUM_PLANES, *PLANE_SHAPE), scalars


try:
    from az import _fast
except ImportError:         # Cython 版をビルドしていない (python setup.py build_ext --inplace)
    _fast = None
else:
    assert all(_fast.CONSTANTS[name] == globals()[name] for name in _fast.CONSTANTS)
    assert pybl.HAND_SIZE == 7 and _INITIAL_DECK == 46 and int(pybl.FIRST) == 0 and int(pybl.SECOND) == 1


def encode(obs: pybl.Observation, flag_masks: list | None = None) -> tuple:
    """観測を (planes, scalars) に変換する. planes は (NUM_PLANES, 6, 10), scalars は (NUM_SCALARS,) の float32.

    flag_masks: flag_card_masks(obs) と同じもの (探索で渡すと, フラッグのカードを obs から取り出す時間を省ける)
    """
    if _fast is not None:
        return _fast.encode(obs, flag_masks)
    return _encode_py(obs, flag_masks)


def legal_mask(obs: pybl.Observation) -> np.ndarray:
    """合法手のマスク ((NUM_ACTIONS,) の bool). 手番でない場合や終局後は全て False"""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[obs.get_legal_moves()] = True
    return mask


class Symmetry:
    """色の入れ替えとフラッグの左右反転による対称変換.

    color_perm[c] は色 c の変換後の色. mirror が True ならフラッグ i を 8 - i に移す.
    変換は全て「元の配列 → 変換後の配列」で, 価値は変換しても変わらない.
    """

    def __init__(self, color_perm=tuple(range(NUM_COLORS)), mirror: bool = False):
        self.color_perm = tuple(int(c) for c in color_perm)
        if sorted(self.color_perm) != list(range(NUM_COLORS)):
            raise ValueError(f"color_perm must be a permutation of 0..{NUM_COLORS - 1}: {color_perm}")
        self.mirror = bool(mirror)

        # 元の番号 -> 変換後の番号
        self.flag_map = np.array([NUM_FLAGS - 1 - i if self.mirror else i for i in range(NUM_FLAGS)])
        self.card_map = np.array([self.color_perm[c // NUM_RANKS] * NUM_RANKS + c % NUM_RANKS
                                  for c in range(NUM_CARDS)])
        self.action_map = np.empty(NUM_ACTIONS, dtype=np.int64)
        for card in range(NUM_CARDS):
            for flag in range(NUM_FLAGS):
                self.action_map[card * NUM_FLAGS + flag] = self.card_map[card] * NUM_FLAGS + self.flag_map[flag]
        self.action_map[PASS_ACTION] = PASS_ACTION
        channel_map = np.arange(NUM_PLANES)
        channel_map[PLANE_MY_FLAG:PLANE_MY_FLAG + NUM_FLAGS] = PLANE_MY_FLAG + self.flag_map
        channel_map[PLANE_OPP_FLAG:PLANE_OPP_FLAG + NUM_FLAGS] = PLANE_OPP_FLAG + self.flag_map
        scalar_map = np.arange(NUM_SCALARS)
        for start in (SCALAR_MY_CLAIMED, SCALAR_OPP_CLAIMED, SCALAR_MY_FIRST, SCALAR_OPP_FIRST):
            scalar_map[start:start + NUM_FLAGS] = start + self.flag_map

        # 変換後の配列を new = old[..., inverse] で求めるための逆写像
        self._card_inv = np.argsort(self.card_map)
        self._action_inv = np.argsort(self.action_map)
        self._channel_inv = np.argsort(channel_map)
        self._scalar_inv = np.argsort(scalar_map)

    def planes(self, planes: np.ndarray) -> np.ndarray:
        """(..., NUM_PLANES, 6, 10) の入力を変換する"""
        lead = planes.shape[:-3]
        flat = planes.reshape(*lead, NUM_PLANES, NUM_CARDS)
        flat = np.take(np.take(flat, self._channel_inv, axis=-2), self._card_inv, axis=-1)
        return flat.reshape(*lead, NUM_PLANES, *PLANE_SHAPE)

    def scalars(self, scalars: np.ndarray) -> np.ndarray:
        """(..., NUM_SCALARS) の入力を変換する"""
        return np.take(scalars, self._scalar_inv, axis=-1)

    def policy(self, policy: np.ndarray) -> np.ndarray:
        """(..., NUM_ACTIONS) の方策 (または合法手のマスク) を変換する"""
        return np.take(policy, self._action_inv, axis=-1)

    def action(self, action: int) -> int:
        """行動の番号を変換する"""
        return int(self.action_map[action])

    def inverse(self) -> "Symmetry":
        inv = [0] * NUM_COLORS
        for c, d in enumerate(self.color_perm):
            inv[d] = c
        return Symmetry(inv, self.mirror)

    def __repr__(self) -> str:
        return f"Symmetry(color_perm={self.color_perm}, mirror={self.mirror})"


IDENTITY = Symmetry()


def all_symmetries() -> list:
    """全ての対称変換 (720 × 2 = 1440 通り)"""
    return [Symmetry(perm, mirror)
            for perm in itertools.permutations(range(NUM_COLORS)) for mirror in (False, True)]


def random_symmetry(rng) -> Symmetry:
    """ランダムな対称変換. rng は random.Random または numpy の Generator"""
    perm = list(range(NUM_COLORS))
    rng.shuffle(perm)
    mirror = bool(rng.random() < 0.5)
    return Symmetry(perm, mirror)


def symmetry_tables() -> dict:
    """全ての対称変換の逆写像の表 (学習中に GPU 上でまとめて変換するために使う).

    返り値の各配列は (1440, 長さ) で, 変換後の配列は new[..., j] = old[..., table[s, j]] で求められる.
        "channel": (1440, NUM_PLANES)   planes のチャンネル
        "card":    (1440, NUM_CARDS)    planes のカード (色 * 10 + 数字 - 1)
        "scalar":  (1440, NUM_SCALARS)  scalars
        "action":  (1440, NUM_ACTIONS)  方策と合法手のマスク
    """
    syms = all_symmetries()
    return {
        "channel": np.stack([s._channel_inv for s in syms]),
        "card": np.stack([s._card_inv for s in syms]),
        "scalar": np.stack([s._scalar_inv for s in syms]),
        "action": np.stack([s._action_inv for s in syms]),
    }
