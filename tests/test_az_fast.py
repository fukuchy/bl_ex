"""az/_fast.pyx (Cython 版の encode・Node・_select) が Python 版と同じ結果になることのテスト"""
import random

import numpy as np
import pytest

import pybl
from az import features as F
from az import mcts

_fast = pytest.importorskip("az._fast", reason="Cython 版をビルドしていない (python setup.py build_ext --inplace)")


def random_states(count: int, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(count):
        state = pybl.GameState(seed=rng.getrandbits(32))
        for _ in range(rng.randrange(80)):
            if state.is_terminal:
                break
            moves = state.get_legal_moves()
            state.play(moves[rng.randrange(len(moves))])
        yield state


def test_cython_is_used():
    assert F._fast is _fast
    assert mcts.Node is _fast.Node and mcts._select is _fast.select


@pytest.mark.parametrize("seed", range(4))
def test_encode_matches_python(seed):
    for state in random_states(50, seed):
        for player in (pybl.FIRST, pybl.SECOND):
            obs = state.observe(player)
            expected = F._encode_py(obs)
            masks = F.flag_card_masks(obs)
            for actual in (_fast.encode(obs), _fast.encode(obs, masks), F._encode_py(obs, masks)):
                assert actual[0].dtype == np.float32 and actual[1].dtype == np.float32
                assert np.array_equal(actual[0], expected[0]) and np.array_equal(actual[1], expected[1])


class RowEvaluator:
    """各局面の入力だけから決まる値を返すダミーの評価器"""

    def __call__(self, planes, scalars):
        flat = planes.reshape(len(planes), -1).astype(np.float32)
        logits = flat[:, :F.NUM_ACTIONS] * 0.7 - flat[:, -F.NUM_ACTIONS:] * 0.3 + scalars[:, :1] * 0.1
        return logits.astype(np.float32), np.tanh(scalars.sum(axis=1) * 0.1 - flat.mean(axis=1)).astype(np.float32)


@pytest.mark.parametrize("fpu_reduction", [None, 0.2])
@pytest.mark.parametrize("seed", range(4))
def test_search_matches_python(seed, fpu_reduction, monkeypatch):
    states = list(random_states(5, 100 + seed))
    kwargs = dict(num_simulations=120, batch_size=8, fpu_reduction=fpu_reduction, dirichlet_alpha=0.3)
    fast = [mcts.search(s.observe(s.side_to_move), RowEvaluator(), rng=random.Random(seed), **kwargs)
            for s in states if not s.is_terminal]
    monkeypatch.setattr(mcts, "_select", mcts._select_py)
    monkeypatch.setattr(mcts, "Node", mcts._PyNode)
    monkeypatch.setattr(F, "_fast", None)
    slow = [mcts.search(s.observe(s.side_to_move), RowEvaluator(), rng=random.Random(seed), **kwargs)
            for s in states if not s.is_terminal]
    for a, b in zip(fast, slow):
        assert a.moves == b.moves
        assert np.array_equal(a.visits, b.visits)
        assert np.allclose(a.q_values, b.q_values)


def test_flag_masks_tracked_in_search(monkeypatch):
    """探索の中で更新するフラッグのカードのマスクが, 葉の局面から求めたものと一致する"""
    original = F.encode
    checked = []

    def checking_encode(obs, flag_masks=None):
        if flag_masks is not None:
            assert flag_masks == F.flag_card_masks(obs)
            checked.append(1)
        return original(obs, flag_masks)

    monkeypatch.setattr(F, "encode", checking_encode)
    for state in random_states(6, 7):
        if not state.is_terminal:
            mcts.search(state.observe(state.side_to_move), RowEvaluator(), num_simulations=60, rng=random.Random(1))
    assert len(checked) > 100
