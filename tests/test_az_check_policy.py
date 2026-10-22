"""az/check_policy.py (方策の正解 π の平らさの統計) のテスト"""
import sys

import numpy as np
import pytest

from az import features as F
from az.check_policy import load_sample, policy_stats


def make_policy(rows: list) -> tuple:
    """各行を {手: 確率} で与えて (policy, masks) を作る. 合法手は確率を持つ手"""
    policy = np.zeros((len(rows), F.NUM_ACTIONS), dtype=np.float16)
    for i, row in enumerate(rows):
        for action, prob in row.items():
            policy[i, action] = prob
    return policy, policy > 0


def test_uniform_and_one_hot():
    uniform, masks = make_policy([{a: 0.25 for a in range(4)}])
    s = policy_stats(uniform, masks)
    assert s["entropy"] == pytest.approx(np.log(4), abs=1e-5)
    assert s["normalized_entropy"] == pytest.approx(1.0, abs=1e-5)
    assert s["effective_moves"] == pytest.approx(4.0, abs=1e-4)
    assert s["max_prob"] == pytest.approx(0.25)

    one_hot, masks = make_policy([{7: 1.0}])
    masks[0, :3] = True                         # 合法手が他にあっても π は1つの手に集中している
    s = policy_stats(one_hot, masks)
    assert s["entropy"] == pytest.approx(0.0, abs=1e-6)
    assert s["effective_moves"] == pytest.approx(1.0)
    assert s["max_prob"] == pytest.approx(1.0)
    assert s["legal"] == 4


def test_flatter_policy_has_higher_entropy():
    sharp, masks = make_policy([{0: 0.9, 1: 0.05, 2: 0.05}, {3: 0.8, 4: 0.2}])
    flat, flat_masks = make_policy([{0: 0.4, 1: 0.3, 2: 0.3}, {3: 0.5, 4: 0.5}])
    assert policy_stats(flat, flat_masks)["entropy"] > policy_stats(sharp, masks)["entropy"]
    assert policy_stats(flat, flat_masks)["max_prob"] < policy_stats(sharp, masks)["max_prob"]


def test_float16_rounding_is_renormalized():
    policy, masks = make_policy([{a: 1 / 3 for a in range(3)}])
    assert policy_stats(policy, masks)["normalized_entropy"] == pytest.approx(1.0, abs=1e-4)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """小さなネットワークの自己対戦で, 2世代分のデータと, 方策の分布を持たないデータを作る"""
    torch = pytest.importorskip("torch")
    from az import selfplay
    from az.model import TokenNet, save_checkpoint
    root = tmp_path_factory.mktemp("run")
    torch.manual_seed(0)
    checkpoint = root / "checkpoints" / "gen_0000.pt"
    checkpoint.parent.mkdir()
    save_checkpoint(checkpoint, TokenNet(d_model=32, layers=1, heads=2))
    config = {**selfplay.DEFAULT_CONFIG, "num_simulations": 16, "batch_size": 4, "temperature_moves": 4}
    for gen in (1, 2):
        selfplay.generate(checkpoint, 3, root / "data" / f"gen_{gen:04d}", config, jobs=1, seed=gen,
                          save_every=1, log=lambda *_: None)
    # 方策の分布を持たない (教師あり学習の) データは飛ばされる
    pretrain = root / "data" / "gen_0003"
    pretrain.mkdir()
    np.savez(pretrain / "part_0000.npz", actions=np.zeros(2, dtype=np.int16))
    return root, checkpoint


def test_load_sample(run):
    root, _ = run
    all_files = load_sample(root / "data" / "gen_0001", max_files=0)
    sample = load_sample(root / "data" / "gen_0001", max_files=2)
    assert len(sample["policy"]) < len(all_files["policy"])
    assert load_sample(root / "data" / "gen_0003", max_files=0) is None
    assert load_sample(root / "missing", max_files=0) is None


def test_network_stats(run):
    from az.check_policy import network_stats
    from az.model import load_checkpoint
    root, checkpoint = run
    data = load_sample(root / "data" / "gen_0001", max_files=0)
    model, _ = load_checkpoint(checkpoint)
    n = network_stats(model, data, batch_size=7)
    s = policy_stats(data["policy"], data["masks"])
    assert n["kl"] >= -1e-6
    assert n["cross_entropy"] == pytest.approx(s["entropy"] + n["kl"], abs=1e-3)
    assert 0 <= n["top1_agree"] <= 1
    assert 0 < n["net_entropy"] <= np.log(F.NUM_ACTIONS)


def test_main(run, monkeypatch, capsys):
    from az import check_policy
    root, checkpoint = run
    monkeypatch.setattr(sys, "argv", ["check_policy", "--run", str(root), "--checkpoint", str(checkpoint),
                                      "--max-files", "0", "--device", "cpu"])
    check_policy.main()
    out = capsys.readouterr().out
    assert "gen_0001" in out and "gen_0002" in out and "飛ばす" in out
    assert "ネットワーク gen_0000" in out
