"""az/selfplay.py (自己対戦によるデータの生成) と az/train.py (学習) のテスト"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pybl
from az import features as F
from az import selfplay
from az.model import TokenNet, load_checkpoint, save_checkpoint
from az.train import fit, load_dataset


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "net.pt"
    torch.manual_seed(0)
    save_checkpoint(path, TokenNet(d_model=32, layers=1, heads=2))
    return str(path)


def small_config():
    return {**selfplay.DEFAULT_CONFIG, "num_simulations": 16, "batch_size": 4, "temperature_moves": 4}


@pytest.mark.parametrize("seed", range(3))
def test_selfplay_game_record(checkpoint, seed):
    record = selfplay.play_selfplay_game((seed, checkpoint, seed, small_config()))
    n = len(record["values"])
    assert n > 10
    assert record["planes"].shape == (n, F.NUM_PLANES, 6, 10) and record["planes"].dtype == np.uint8
    assert record["policy"].shape == (n, F.NUM_ACTIONS) and record["policy"].dtype == np.float16
    policy = record["policy"].astype(np.float32)
    assert np.allclose(policy.sum(1), 1.0, atol=1e-2)
    assert np.all(policy[~record["masks"]] == 0)              # 合法手以外に確率は無い
    assert np.all(record["masks"].sum(1) > 1)                # 選択の余地が無い局面は記録しない
    assert np.all(np.abs(record["q"]) <= 1 + 1e-6)
    # 勝敗の向き: 勝者の局面は +1, 敗者の局面は -1 (引き分けは 0)
    winner = int(record["winner"][0])
    first = record["scalars"][:, F.SCALAR_FIRST_PLAYER]
    if winner == pybl.NULL_PLAYER:
        assert np.all(record["values"] == 0)
    else:
        assert set(np.unique(record["values"])) <= {-1.0, 1.0}
        # 同じプレイヤーの局面は同じ値, 先手と後手の局面は逆の値になる
        assert len(set(record["values"][first == 1])) <= 1 and len(set(record["values"][first == 0])) <= 1
        if (first == 1).any() and (first == 0).any():
            assert record["values"][first == 1][0] == -record["values"][first == 0][0]


def test_generate_and_fit(checkpoint, tmp_path):
    out = tmp_path / "data"
    stats = selfplay.generate(checkpoint, 4, out, small_config(), jobs=1, seed=0, save_every=2, log=lambda *_: None)
    assert stats["games"] == 4 and len(list(out.glob("*.npz"))) == 2
    data = load_dataset([out])
    assert len(data["values"]) == stats["positions"] and "policy" in data and "q" in data
    model, _ = load_checkpoint(checkpoint)
    new = tmp_path / "new.pt"
    result = fit(model, data, out=new, epochs=1, batch_size=64, device="cpu", keep_best=False, val_ratio=0.25,
                 q_weight=0.5, log=lambda *_: None)
    assert new.exists() and result["val"] is not None
    loaded, ckpt = load_checkpoint(new)
    assert ckpt["epoch"] == 1
