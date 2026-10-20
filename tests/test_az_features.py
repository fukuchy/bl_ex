"""az/features.py (入出力のエンコードと対称変換) のテスト"""
import random

import numpy as np
import pytest

import pybl
from az import features as F


def random_state(seed: int, max_moves: int = 40) -> pybl.GameState:
    """ランダムに進めた途中局面 (終局していない)"""
    rng = random.Random(seed)
    while True:
        state = pybl.GameState(seed=rng.getrandbits(32))
        for _ in range(rng.randrange(max_moves)):
            if state.is_terminal:
                break
            moves = state.get_legal_moves()
            state.play(moves[rng.randrange(len(moves))])
        if not state.is_terminal:
            return state


def transformed_state(state: pybl.GameState, sym: F.Symmetry) -> pybl.GameState:
    """局面そのものに対称変換を施した局面を作る (カードの色とフラッグの位置を入れ替える)"""
    card = lambda c: int(sym.card_map[int(c)])
    players = (pybl.FIRST, pybl.SECOND)
    hands = tuple([card(c) for c in pybl.CardIterator(state.get_hand(p))] for p in players)
    flags = [None] * F.NUM_FLAGS
    owners = [None] * F.NUM_FLAGS
    completers = [None] * F.NUM_FLAGS
    for i in range(F.NUM_FLAGS):
        j = int(sym.flag_map[i])
        flags[j] = tuple([card(c) for c in state.get_flag_cards(i, p)] for p in players)
        owners[j] = state.get_flag_owner(i)
        completers[j] = state.get_flag_first_completer(i)
    used = {c for h in hands for c in h} | {c for f in flags for side in f for c in side}
    deck = [c for c in range(F.NUM_CARDS) if c not in used]   # 山札の順番はエンコードに影響しない
    new = pybl.GameState(seed=0)
    new.set_position(state.side_to_move, hands, flags, deck, owners=owners, first_completers=completers,
                     first_player=state.first_player)
    return new


def test_constants():
    assert F.NUM_ACTIONS == 541 and F.PASS_ACTION == 540
    assert F.NUM_PLANES == 20 and F.NUM_SCALARS == 41


@pytest.mark.parametrize("seed", range(30))
def test_every_card_has_exactly_one_location(seed):
    state = random_state(seed)
    for p in (pybl.FIRST, pybl.SECOND):
        planes, scalars = F.encode(state.observe(p))
        assert planes.shape == (F.NUM_PLANES, 6, 10) and planes.dtype == np.float32
        assert scalars.shape == (F.NUM_SCALARS,) and scalars.dtype == np.float32
        assert np.array_equal(planes.sum(axis=0), np.ones((6, 10)))


def test_encode_contents():
    state = random_state(3)
    me = state.side_to_move
    opp = pybl.to_opponent(me)
    obs = state.observe(me)
    planes, scalars = F.encode(obs)
    flat = planes.reshape(F.NUM_PLANES, F.NUM_CARDS)
    assert set(np.flatnonzero(flat[F.PLANE_HAND])) == {int(c) for c in pybl.CardIterator(obs.hand)}
    assert set(np.flatnonzero(flat[F.PLANE_UNSEEN])) == {int(c) for c in pybl.CardIterator(obs.unseen_cards)}
    for i in range(F.NUM_FLAGS):
        assert set(np.flatnonzero(flat[F.PLANE_MY_FLAG + i])) == {int(c) for c in obs.get_flag_cards(i, me)}
        assert set(np.flatnonzero(flat[F.PLANE_OPP_FLAG + i])) == {int(c) for c in obs.get_flag_cards(i, opp)}
        assert scalars[F.SCALAR_MY_CLAIMED + i] == (obs.get_flag_owner(i) == me)
        assert scalars[F.SCALAR_OPP_CLAIMED + i] == (obs.get_flag_owner(i) == opp)
    assert scalars[F.SCALAR_DECK] == pytest.approx(obs.deck_count / 46)
    assert scalars[F.SCALAR_OPP_HAND] == pytest.approx(obs.opponent_hand_count / 7)


@pytest.mark.parametrize("seed", range(20))
def test_legal_mask(seed):
    state = random_state(seed)
    obs = state.observe(state.side_to_move)
    mask = F.legal_mask(obs)
    assert mask.shape == (F.NUM_ACTIONS,) and mask.dtype == bool
    assert set(np.flatnonzero(mask)) == set(obs.get_legal_moves().tolist())
    assert not F.legal_mask(state.observe(pybl.to_opponent(state.side_to_move))).any()


def test_all_symmetries_are_distinct():
    syms = F.all_symmetries()
    assert len(syms) == 1440
    assert len({tuple(s.action_map) for s in syms}) == 1440


@pytest.mark.parametrize("seed", range(40))
def test_symmetry_matches_transformed_state(seed):
    """局面を実際に変換してエンコードしたものと, エンコードしたものを変換したものが一致する"""
    rng = random.Random(seed)
    state = random_state(seed)
    sym = F.random_symmetry(rng)
    if seed % 4 == 0:
        sym = F.Symmetry(sym.color_perm, mirror=True)
    new = transformed_state(state, sym)
    for p in (pybl.FIRST, pybl.SECOND):
        planes, scalars = F.encode(state.observe(p))
        new_planes, new_scalars = F.encode(new.observe(p))
        assert np.array_equal(sym.planes(planes), new_planes)
        # set_position は直前のパスの情報を引き継がないので, その項目以外を比べる
        keep = np.arange(F.NUM_SCALARS) != F.SCALAR_PASSED
        assert np.array_equal(sym.scalars(scalars)[keep], new_scalars[keep])
    me = state.side_to_move
    assert np.array_equal(sym.policy(F.legal_mask(state.observe(me))), F.legal_mask(new.observe(me)))


@pytest.mark.parametrize("seed", range(20))
def test_symmetry_actions_follow_the_rules(seed):
    """変換した手を変換した局面で指すと, 元の手を元の局面で指して変換したものと一致する"""
    rng = random.Random(seed)
    state = random_state(seed)
    sym = F.random_symmetry(rng)
    new = transformed_state(state, sym)
    me = state.side_to_move
    for move in state.get_legal_moves().tolist()[:10]:
        a, b = state.copy(), new.copy()
        claimed_a = a.play(move)
        claimed_b = b.play(sym.action(move))
        assert claimed_b == sum(1 << int(sym.flag_map[i]) for i in range(F.NUM_FLAGS) if claimed_a >> i & 1)
        assert a.result == b.result
        # 補充で引くカードは山札の順番に依存するので, 盤面 (planes の手札以外) と確保の情報を比べる
        pa, sa = F.encode(a.observe(me))
        pb, sb = F.encode(b.observe(me))
        board = [ch for ch in range(F.NUM_PLANES) if ch not in (F.PLANE_HAND, F.PLANE_UNSEEN)]
        assert np.array_equal(sym.planes(pa)[board], pb[board])
        assert np.array_equal(sym.scalars(sa)[:F.SCALAR_DECK], sb[:F.SCALAR_DECK])


@pytest.mark.parametrize("seed", range(10))
def test_inverse_and_batch(seed):
    rng = random.Random(seed)
    sym = F.random_symmetry(rng)
    inv = sym.inverse()
    planes = np.stack([F.encode(random_state(seed * 10 + k).observe(pybl.FIRST))[0] for k in range(4)])
    scalars = np.random.default_rng(seed).random((4, F.NUM_SCALARS)).astype(np.float32)
    policy = np.random.default_rng(seed).random((4, F.NUM_ACTIONS)).astype(np.float32)
    # バッチの各要素を個別に変換した結果と一致する
    assert np.array_equal(sym.planes(planes), np.stack([sym.planes(x) for x in planes]))
    # 逆変換で元に戻る
    assert np.array_equal(inv.planes(sym.planes(planes)), planes)
    assert np.array_equal(inv.scalars(sym.scalars(scalars)), scalars)
    assert np.array_equal(inv.policy(sym.policy(policy)), policy)
    assert all(inv.action(sym.action(a)) == a for a in range(F.NUM_ACTIONS))
    # 方策の変換と行動の変換が対応する
    a = rng.randrange(F.NUM_ACTIONS)
    onehot = np.zeros(F.NUM_ACTIONS); onehot[a] = 1
    assert np.flatnonzero(sym.policy(onehot)).tolist() == [sym.action(a)]
