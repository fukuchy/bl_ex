"""az/model.py (ネットワーク・損失・対称変換によるデータ拡張・保存と読み込み) のテスト"""
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pybl
from az import features as F
from az.model import BattleLineNet, SymmetryAugmenter, TokenNet, build_model, load_checkpoint, loss_fn, \
    masked_log_softmax, predict, save_checkpoint

ARCHS = [lambda: BattleLineNet(hidden=64, blocks=2), lambda: TokenNet(d_model=32, layers=2, heads=2)]


def sample_batch(n: int, seed: int = 0):
    rng = random.Random(seed)
    planes, scalars, masks = [], [], []
    while len(planes) < n:
        state = pybl.GameState(seed=rng.getrandbits(32))
        for _ in range(rng.randrange(30)):
            if state.is_terminal:
                break
            moves = state.get_legal_moves()
            state.play(moves[rng.randrange(len(moves))])
        if state.is_terminal:
            continue
        obs = state.observe(state.side_to_move)
        p, s = F.encode(obs)
        planes.append(p); scalars.append(s); masks.append(F.legal_mask(obs))
    return np.stack(planes), np.stack(scalars), np.stack(masks)


@pytest.mark.parametrize("make", ARCHS)
def test_forward_shapes_and_masked_policy(make):
    planes, scalars, masks = sample_batch(8)
    model = make()
    logits, value = model(torch.as_tensor(planes), torch.as_tensor(scalars))
    assert logits.shape == (8, F.NUM_ACTIONS) and value.shape == (8,)
    assert torch.all(value.abs() <= 1)
    probs, values = predict(model, planes, scalars, masks)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)
    assert np.all(probs[~masks] < 1e-12)          # 合法手以外の確率は 0
    assert values.shape == (8,)


@pytest.mark.parametrize("make", [lambda: BattleLineNet(hidden=128, blocks=2),
                                  lambda: TokenNet(d_model=64, layers=2, heads=2)])
def test_loss_decreases_on_fixed_batch(make):
    planes, scalars, masks = sample_batch(32, seed=1)
    p, s, m = torch.as_tensor(planes), torch.as_tensor(scalars), torch.as_tensor(masks)
    rng = np.random.default_rng(0)
    actions = torch.as_tensor([rng.choice(np.flatnonzero(row)) for row in masks])
    policy = torch.zeros(32, F.NUM_ACTIONS).scatter_(1, actions[:, None], 1.0)
    values = torch.as_tensor(rng.choice([-1.0, 1.0], size=32), dtype=torch.float32)
    torch.manual_seed(0)
    model = make()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = None
    for _ in range(300):
        logits, value = model(p, s)
        loss, _, _ = loss_fn(logits, value, policy, values, m)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first * 0.1         # 小さなバッチなら丸暗記できる (実装の誤りの検出)
    assert (masked_log_softmax(logits, m).argmax(1) == actions).float().mean() == 1.0


def test_augmenter_matches_numpy_symmetry():
    """GPU 上でのまとめての変換が, 要素ごとの Symmetry による変換と一致する"""
    planes, scalars, masks = sample_batch(16, seed=2)
    policy = np.random.default_rng(0).random((16, F.NUM_ACTIONS)).astype(np.float32)
    augment = SymmetryAugmenter("cpu")
    gen = torch.Generator().manual_seed(123)
    out = augment(torch.as_tensor(planes), torch.as_tensor(scalars), torch.as_tensor(policy),
                  torch.as_tensor(masks), generator=gen)
    sym_ids = torch.randint(augment.count, (16,), generator=torch.Generator().manual_seed(123)).tolist()
    syms = F.all_symmetries()
    for i, k in enumerate(sym_ids):
        sym = syms[k]
        assert np.array_equal(out[0][i].numpy(), sym.planes(planes[i]))
        assert np.array_equal(out[1][i].numpy(), sym.scalars(scalars[i]))
        assert np.array_equal(out[2][i].numpy(), sym.policy(policy[i]))
        assert np.array_equal(out[3][i].numpy(), sym.policy(masks[i]))


@pytest.mark.parametrize("make", ARCHS)
def test_checkpoint_roundtrip(tmp_path, make):
    planes, scalars, masks = sample_batch(4, seed=3)
    model = make().eval()
    path = tmp_path / "net.pt"
    save_checkpoint(path, model, epoch=5)
    loaded, checkpoint = load_checkpoint(path)
    assert checkpoint["epoch"] == 5 and checkpoint["config"] == model.config
    assert type(loaded) is type(model)
    a = predict(model, planes, scalars, masks)
    b = predict(loaded, planes, scalars, masks)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_token_policy_order_matches_actions():
    """TokenNet の方策のロジットの並び (カード * 9 + フラッグ, 最後がパス) が行動の番号と一致する"""
    planes, scalars, masks = sample_batch(2, seed=4)
    model = TokenNet(d_model=32, layers=1, heads=2).eval()
    p, s = torch.as_tensor(planes), torch.as_tensor(scalars)
    with torch.no_grad():
        logits, _ = model(p, s)
        h = model.norm(model.encoder(model.tokens(p, s)))
        card = model.policy_card(h[:, :F.NUM_CARDS]); flag = model.policy_flag(h[:, F.NUM_CARDS:F.NUM_CARDS + F.NUM_FLAGS])
    for c, f in [(0, 0), (13, 4), (59, 8)]:
        expected = (card[:, c] * flag[:, f]).sum(-1) / (32 ** 0.5)
        assert torch.allclose(logits[:, c * F.NUM_FLAGS + f], expected, atol=1e-5)


def test_build_model():
    assert isinstance(build_model({"arch": "token", "d_model": 32, "layers": 1, "heads": 2}), TokenNet)
    assert isinstance(build_model({"hidden": 32, "blocks": 1}), BattleLineNet)     # 旧形式 (arch なし)
