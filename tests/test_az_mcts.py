"""az/mcts.py (ネットワークを使う ISMCTS) のテスト"""
import random

import numpy as np
import pytest

import pybl
from az import features as F
from az import mcts


class UniformEvaluator:
    """方策は一様 (ロジット 0), 価値は 0 を返すダミーの評価器. 呼び出しのバッチの大きさを記録する"""

    def __init__(self):
        self.batch_sizes = []

    def __call__(self, planes, scalars):
        self.batch_sizes.append(len(planes))
        return np.zeros((len(planes), F.NUM_ACTIONS), dtype=np.float32), np.zeros(len(planes), dtype=np.float32)


def random_position(seed: int, max_moves: int = 30) -> pybl.GameState:
    rng = random.Random(seed)
    while True:
        state = pybl.GameState(seed=rng.getrandbits(32))
        for _ in range(rng.randrange(max_moves)):
            if state.is_terminal:
                break
            moves = state.get_legal_moves()
            state.play(moves[rng.randrange(len(moves))])
        if not state.is_terminal and len(state.get_legal_moves()) > 1:
            return state


def walk(node):
    yield node
    for child in node.children.values():
        yield from walk(child)


@pytest.mark.parametrize("seed", range(6))
def test_visit_counts_and_virtual_loss(seed, monkeypatch):
    state = random_position(seed)
    obs = state.observe(state.side_to_move)
    roots = []
    original = mcts.Node

    class RecordingNode(original):
        __slots__ = ()

        def __init__(self, move, player, parent):
            super().__init__(move, player, parent)
            if parent is None:
                roots.append(self)

    monkeypatch.setattr(mcts, "Node", RecordingNode)
    evaluator = UniformEvaluator()
    result = mcts.search(obs, evaluator, num_simulations=150, batch_size=8, rng=random.Random(seed))
    assert result.visits.sum() == 150
    root = roots[0]
    for node in walk(root):
        if node is not root:
            # 仮想損失が取り除かれていれば, 価値の平均は -1〜1 に収まる
            assert node.visits >= 1 and abs(node.value_sum) <= node.visits + 1e-9
        assert sum(c.visits for c in node.children.values()) <= max(node.visits, 150)
    assert max(evaluator.batch_sizes) <= 8


@pytest.mark.parametrize("seed", range(6))
def test_policy_target(seed):
    state = random_position(seed)
    obs = state.observe(state.side_to_move)
    result = mcts.search(obs, UniformEvaluator(), num_simulations=100, rng=random.Random(seed))
    legal = set(obs.get_legal_moves().tolist())
    assert result.policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert all(result.policy[a] == 0 for a in range(F.NUM_ACTIONS) if a not in legal)
    # 代表にまとめた手は, 同等な手に均等に分けられる
    classes = mcts._equivalence_classes(obs, True)
    for rep, members in classes.items():
        values = result.policy[members]
        assert np.allclose(values, values[0])


def test_initial_position_uses_symmetry():
    state = pybl.GameState(seed=0)
    obs = state.observe(state.side_to_move)
    result = mcts.search(obs, UniformEvaluator(), num_simulations=50, rng=random.Random(0))
    assert len(result.moves) == 35                       # 63 手が左右対称で 35 手にまとまる
    full = mcts.search(obs, UniformEvaluator(), num_simulations=50, use_representatives=False, rng=random.Random(0))
    assert len(full.moves) == 63


def test_reproducible():
    state = random_position(7)
    obs = state.observe(state.side_to_move)
    a = mcts.search(obs, UniformEvaluator(), num_simulations=80, rng=random.Random(3))
    b = mcts.search(obs, UniformEvaluator(), num_simulations=80, rng=random.Random(3))
    assert a.best_move() == b.best_move() and np.array_equal(a.visits, b.visits)


def find_winning_positions(count: int):
    """その手を指せば勝ちが決まる手と, そうでない手の両方がある局面を集める"""
    found, seed = [], 0
    while len(found) < count and seed < 3000:
        rng = random.Random(seed)
        state = pybl.GameState(seed=seed)
        seed += 1
        while not state.is_terminal:
            me = state.side_to_move
            win = pybl.FIRST_WIN if me == pybl.FIRST else pybl.SECOND_WIN
            moves = state.get_legal_moves().tolist()
            winning = []
            for m in moves:
                s = state.copy(); s.play(m)
                if s.result == win:
                    winning.append(m)
            if winning and len(winning) < len(moves):
                found.append((state.copy(), set(winning)))
                break
            state.play(moves[rng.randrange(len(moves))])
    return found


@pytest.mark.parametrize("case", range(5))
def test_finds_immediate_win(case):
    positions = find_winning_positions(5)
    state, winning = positions[case]
    obs = state.observe(state.side_to_move)
    result = mcts.search(obs, UniformEvaluator(), num_simulations=300, use_representatives=False,
                         rng=random.Random(case))
    assert result.best_move() in winning


def test_az_move_with_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from az.model import TokenNet, save_checkpoint
    import agent_az
    path = tmp_path / "net.pt"
    torch.manual_seed(0)
    save_checkpoint(path, TokenNet(d_model=32, layers=1, heads=2))
    rng = random.Random(0)
    state = pybl.GameState(seed=1)
    while not state.is_terminal:
        obs = state.observe(state.side_to_move)
        move = agent_az.az_move(obs, checkpoint=str(path), num_simulations=20, batch_size=4, rng=rng)
        assert state.is_legal(move)
        state.play(move)
