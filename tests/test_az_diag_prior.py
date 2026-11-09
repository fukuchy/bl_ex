"""az/diag_prior.py (事前確率の使い回しの診断) と, 自己対戦・ループへの prior_cache の受け渡しのテスト"""
import numpy as np
import pytest

from az import features as F
from az import selfplay
from az.diag_prior import format_stats
from az.mcts import PriorStats


class RowEvaluator:
    """各局面の入力だけから決まる値を返すダミーの評価器. 評価した局面の数を数える"""

    def __init__(self):
        self.rows = 0

    def __call__(self, planes, scalars):
        self.rows += len(planes)
        flat = planes.reshape(len(planes), -1).astype(np.float32)
        logits = flat[:, :F.NUM_ACTIONS] * 0.7 - flat[:, -F.NUM_ACTIONS:] * 0.3 + scalars[:, :1] * 0.1
        return logits.astype(np.float32), np.tanh(scalars.sum(axis=1) * 0.1 - flat.mean(axis=1)).astype(np.float32)


@pytest.mark.parametrize("prior_cache", ["none", "hand"])
def test_selfplay_with_stats(prior_cache):
    config = {**selfplay.DEFAULT_CONFIG, "num_simulations": 32, "batch_size": 8, "c_puct": 1.0,
              "fpu_reduction": 0.2, "prior_cache": prior_cache}
    stats, evaluator = PriorStats(), RowEvaluator()
    records = list(selfplay.play_games_batched([(i, i, config, stats) for i in range(3)], evaluator, 3))
    assert len(records) == 3
    leaves = stats.leaves
    assert leaves["eval"] + leaves["terminal"] + leaves["collision"] == stats.searches * 32
    assert evaluator.rows == leaves["eval"] + stats.searches          # ルートの評価が探索ごとに1回
    assert stats.select[(1, False)][0] > 0
    text = format_stats(stats)
    assert "手札の不一致" in text and ("まだ無かった" in text) == (prior_cache == "hand")
    if prior_cache == "hand":
        assert leaves["collision"] == 0 and all(row[1] == 0 for row in stats.select.values())


def test_prior_cache_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    selfplay.add_config_arguments(parser)
    assert parser.parse_args([]).prior_cache == "none"
    assert parser.parse_args(["--prior-cache", "hand"]).prior_cache == "hand"
    with pytest.raises(SystemExit):
        parser.parse_args(["--prior-cache", "world"])


def test_loop_spec_passes_prior_cache():
    from az.loop import _az_spec
    config = {**selfplay.DEFAULT_CONFIG, "fpu_reduction": 0.2}
    assert "prior_cache" not in _az_spec("a.pt", config, 200)
    assert _az_spec("a.pt", {**config, "prior_cache": "hand"}, 200).endswith(",prior_cache=hand")
