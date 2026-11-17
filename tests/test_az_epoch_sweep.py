"""az/epoch_sweep.py (事前学習のエポック数の検証) のテスト"""
import csv
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from az import epoch_sweep, pretrain
from az.train import concat, load_dataset


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """random を教師にした小さな棋譜 (2 ファイル)"""
    out = tmp_path_factory.mktemp("pretrain")
    records = [pretrain.play_teacher_game((i, "random", 100 + i, 0.0)) for i in range(12)]
    np.savez_compressed(out / "part_0000.npz", **concat(records[:6]))
    np.savez_compressed(out / "part_0001.npz", **concat(records[6:]))
    return out


def test_parse_run():
    assert epoch_sweep.parse_run("30000x30") == (30000, 30)
    assert epoch_sweep.parse_run("allx10") == (None, 10)


def test_split_games_is_disjoint_and_fixed(data_dir):
    data = load_dataset([data_dir])
    test, train_ids = epoch_sweep.split_games(data, 3, seed=0)
    assert len(np.unique(test["games"])) == 3 and len(train_ids) == 9
    assert not set(np.unique(test["games"])) & set(train_ids)
    again, again_ids = epoch_sweep.split_games(data, 3, seed=0)
    assert np.array_equal(again["games"], test["games"]) and np.array_equal(again_ids, train_ids)


def test_main_writes_results(data_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "sweep"
    monkeypatch.setattr(sys, "argv", [
        "epoch_sweep", "--data", str(data_dir), "--runs", "4x2", "allx1", "--test-games", "3", "--out", str(out),
        "--d-model", "32", "--layers", "1", "--heads", "2", "--batch-size", "64", "--device", "cpu",
        "--val-ratio", "0.25"])
    epoch_sweep.main()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    first, second = summary["results"]
    assert (first["games"], first["epochs"], len(first["rows"])) == (4, 2, 2)
    assert (second["games"], second["epochs"], len(second["rows"])) == (9, 1, 1)
    assert set(first["best_epoch"]) == {"loss", "policy", "value", "acc"}
    for row in first["rows"]:
        assert np.isclose(row["test_loss"], 0.25 * row["test_value"] + row["test_policy"])
        assert 0 <= row["test_acc"] <= 100 and row["lr"] >= 0
    with open(out / "run_4x2.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2
    assert "最小の損失" in capsys.readouterr().out
