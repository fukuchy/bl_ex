# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""自己対戦の探索で何度も呼ばれる処理の Cython 版 (az/features.py の encode と az/mcts.py の Node・_select).

ビルド (bl_ex のディレクトリで):
    python setup.py build_ext --inplace

ビルドしていなければ, features.py と mcts.py は Python 版を使う (結果は同じ).
"""
import numpy as np

from cpython.dict cimport PyDict_GetItem
from cpython.ref cimport PyObject
from libc.math cimport exp, sqrt, INFINITY

# features.py の定数と同じ値 (features.py で一致を確認する)
cdef enum:
    NUM_CARDS = 60
    NUM_FLAGS = 9
    NUM_PLANES = 20
    NUM_SCALARS = 41
    PLANE_HAND = 0
    PLANE_UNSEEN = 1
    PLANE_MY_FLAG = 2
    PLANE_OPP_FLAG = 11
    SCALAR_MY_CLAIMED = 0
    SCALAR_OPP_CLAIMED = 9
    SCALAR_MY_FIRST = 18
    SCALAR_OPP_FIRST = 27
    SCALAR_DECK = 36
    SCALAR_OPP_HAND = 37
    SCALAR_MY_HAND = 38
    SCALAR_PASSED = 39
    SCALAR_FIRST_PLAYER = 40
    HAND_SIZE = 7
    INITIAL_DECK = 46
    MAX_MOVES = 541

CONSTANTS = {
    "NUM_CARDS": NUM_CARDS, "NUM_FLAGS": NUM_FLAGS, "NUM_PLANES": NUM_PLANES, "NUM_SCALARS": NUM_SCALARS,
    "PLANE_HAND": PLANE_HAND, "PLANE_UNSEEN": PLANE_UNSEEN, "PLANE_MY_FLAG": PLANE_MY_FLAG,
    "PLANE_OPP_FLAG": PLANE_OPP_FLAG, "SCALAR_MY_CLAIMED": SCALAR_MY_CLAIMED,
    "SCALAR_OPP_CLAIMED": SCALAR_OPP_CLAIMED, "SCALAR_MY_FIRST": SCALAR_MY_FIRST,
    "SCALAR_OPP_FIRST": SCALAR_OPP_FIRST, "SCALAR_DECK": SCALAR_DECK, "SCALAR_OPP_HAND": SCALAR_OPP_HAND,
    "SCALAR_MY_HAND": SCALAR_MY_HAND, "SCALAR_PASSED": SCALAR_PASSED, "SCALAR_FIRST_PLAYER": SCALAR_FIRST_PLAYER,
}


# ---------------------------------------------------------------------------
# 入力のエンコード
# ---------------------------------------------------------------------------

cdef inline void _fill(float[:, ::1] planes, int channel, unsigned long long bits) noexcept nogil:
    cdef int c
    for c in range(NUM_CARDS):
        if (bits >> c) & 1:
            planes[channel, c] = 1.0


def encode(obs, flag_masks=None):
    """features.encode と同じ. flag_masks は features.flag_card_masks の形 (省略時は obs から求める)"""
    cdef int me = int(obs.player)
    cdef int opp = 1 - me
    cdef int i, completer
    cdef unsigned int my_claimed, opp_claimed
    planes = np.zeros((NUM_PLANES, NUM_CARDS), dtype=np.float32)
    scalars = np.zeros(NUM_SCALARS, dtype=np.float32)
    cdef float[:, ::1] p = planes
    cdef float[::1] s = scalars

    if flag_masks is None:
        from az.features import flag_card_masks
        flag_masks = flag_card_masks(obs)
    _fill(p, PLANE_HAND, int(obs.hand))
    _fill(p, PLANE_UNSEEN, int(obs.unseen_cards))
    for i in range(NUM_FLAGS):
        _fill(p, PLANE_MY_FLAG + i, flag_masks[me * NUM_FLAGS + i])
        _fill(p, PLANE_OPP_FLAG + i, flag_masks[opp * NUM_FLAGS + i])

    my_claimed = obs.get_claimed_flags(me)
    opp_claimed = obs.get_claimed_flags(opp)
    for i in range(NUM_FLAGS):
        s[SCALAR_MY_CLAIMED + i] = (my_claimed >> i) & 1
        s[SCALAR_OPP_CLAIMED + i] = (opp_claimed >> i) & 1
        completer = int(obs.get_flag_first_completer(i))
        s[SCALAR_MY_FIRST + i] = completer == me
        s[SCALAR_OPP_FIRST + i] = completer == opp

    s[SCALAR_DECK] = <float>(<double>obs.deck_count / INITIAL_DECK)
    s[SCALAR_OPP_HAND] = <float>(<double>obs.opponent_hand_count / HAND_SIZE)
    s[SCALAR_MY_HAND] = <float>(<double>obs.get_hand_count(me) / HAND_SIZE)
    s[SCALAR_PASSED] = obs.consecutive_pass_count > 0
    s[SCALAR_FIRST_PLAYER] = int(obs.first_player) == me
    return planes.reshape(NUM_PLANES, 6, 10), scalars


# ---------------------------------------------------------------------------
# 探索の木
# ---------------------------------------------------------------------------

cdef class Node:
    """mcts.py の Node と同じ (属性を C の型で持つ)"""
    cdef public object move
    cdef public object player
    cdef public object parent
    cdef public dict children
    cdef public long visits
    cdef public double value_sum
    cdef public object logits

    def __init__(self, move, player, parent):
        self.move = move
        self.player = player
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.logits = None


def select(Node node, Node root, list root_moves, const double[::1] root_priors, work, double c_puct, double fpu,
           fpu_reduction=None):
    """mcts._select と同じ (PUCT で手を選ぶ). node.logits は float32 の配列"""
    cdef Py_ssize_t n, i
    cdef double priors[MAX_MOVES]
    cdef PyObject* kids[MAX_MOVES]
    cdef int moves[MAX_MOVES]
    cdef const short[:] legal
    cdef const float[:] logits
    cdef double top, total_exp, value_total = 0.0, visited_prior = 0.0, sqrt_total, score, best_score
    cdef long total = 0
    cdef Node child
    cdef int best
    cdef dict children = node.children

    if node is root:
        n = len(root_moves)
        for i in range(n):
            moves[i] = root_moves[i]
            priors[i] = root_priors[i]
    else:
        legal = work.get_legal_moves()
        logits = node.logits
        n = legal.shape[0]
        top = -INFINITY
        for i in range(n):
            moves[i] = legal[i]
            if logits[moves[i]] > top:
                top = logits[moves[i]]
        total_exp = 0.0
        for i in range(n):
            priors[i] = exp(<double>logits[moves[i]] - top)
            total_exp += priors[i]
        for i in range(n):
            priors[i] /= total_exp

    for i in range(n):
        kids[i] = PyDict_GetItem(children, moves[i])
        if kids[i] is not NULL:
            child = <Node>kids[i]
            if child.visits > 0:
                total += child.visits
                value_total += child.value_sum
                visited_prior += priors[i]
    sqrt_total = sqrt(<double>(total if total > 1 else 1))
    if fpu_reduction is not None:
        # 子の価値はその手を指したプレイヤー (このノードの手番) から見た値なので, その平均を親の価値とする
        fpu = (value_total / total if total > 0 else 0.0) - <double>fpu_reduction * sqrt(visited_prior)

    best, best_score = moves[0], -INFINITY
    for i in range(n):
        if kids[i] is NULL or (<Node>kids[i]).visits == 0:
            score = fpu + c_puct * priors[i] * sqrt_total
        else:
            child = <Node>kids[i]
            score = child.value_sum / child.visits + c_puct * priors[i] * sqrt_total / (1 + child.visits)
        if score > best_score:
            best, best_score = moves[i], score
    return best
