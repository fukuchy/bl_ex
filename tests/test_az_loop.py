"""az/loop.py (強化学習のループ) のテスト"""
import csv
import json
import sys

import pytest

torch = pytest.importorskip("torch")

from az import loop
from az.model import TokenNet, save_checkpoint


@pytest.mark.parametrize("agree, expected", [
    (0.90, 0.15),       # 探索がネットの手をなぞるだけ → 下げる
    (0.74, 0.20),       # 目標 0.70 ± 0.05 の中 → そのまま
    (0.66, 0.20),
    (0.60, 0.25),       # 探索がネットの手を覆しすぎ → 上げる
])
def test_adjust_fpu_reduction(agree, expected):
    assert loop.adjust_fpu_reduction(0.2, agree, 0.70, 0.05, 0.05, 0.0, 0.4) == pytest.approx(expected)


def test_adjust_fpu_reduction_is_clamped():
    assert loop.adjust_fpu_reduction(0.02, 0.95, 0.70, 0.05, 0.05, 0.0, 0.4) == 0.0
    assert loop.adjust_fpu_reduction(0.0, 0.95, 0.70, 0.05, 0.05, 0.0, 0.4) == 0.0
    assert loop.adjust_fpu_reduction(0.38, 0.30, 0.70, 0.05, 0.05, 0.0, 0.4) == 0.4


def test_append_csv_merges_new_columns(tmp_path):
    path = tmp_path / "log.csv"
    loop._append_csv(path, {"generation": 1, "best": 0})
    loop._append_csv(path, {"generation": 2, "prior_agree": 0.8, "best": 1})
    loop._append_csv(path, {"generation": 3, "prior_agree": 0.7, "best": 1})
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames == ["generation", "prior_agree", "best"]
    assert [r["generation"] for r in rows] == ["1", "2", "3"]
    assert rows[0]["prior_agree"] == "" and rows[0]["best"] == "0"
    assert rows[2]["prior_agree"] == "0.7"


def test_loop_with_fpu_auto(tmp_path, monkeypatch):
    """--fpu-auto で 2 世代回すと, prior_agree が記録され, fpu_reduction が state.json に保存される"""
    init = tmp_path / "init.pt"
    torch.manual_seed(0)
    save_checkpoint(init, TokenNet(d_model=16, layers=1, heads=2))
    run = tmp_path / "run"
    argv = ["loop", "--run", str(run), "--init", str(init), "--iterations", "2", "--games", "2", "--epochs", "1",
            "--gate-threshold", "0", "--eval-agents", "-j", "1", "--device", "cpu", "--selfplay-device", "cpu",
            "--num-simulations", "8", "--batch-size", "4", "--fpu-auto", "--fpu-reduction", "0.2",
            "--fpu-target", "0.5", "--fpu-band", "0.0", "--fpu-step", "0.1"]
    monkeypatch.setattr(sys, "argv", argv)
    loop.main()
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    with open(run / "log.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert state["generation"] == 2 and len(rows) == 2
    # 1 世代目は 0.2 で自己対戦し, prior_agree が 0.5 からずれた向きに 0.1 ずつ動く
    assert float(rows[0]["fpu_reduction"]) == pytest.approx(0.2)
    for row, nxt in zip(rows, rows[1:] + [None]):
        agree, used = float(row["prior_agree"]), float(row["fpu_reduction"])
        expected = loop.adjust_fpu_reduction(used, agree, 0.5, 0.0, 0.1, 0.0, 0.4)
        assert float(nxt["fpu_reduction"] if nxt else state["fpu_reduction"]) == pytest.approx(expected)
