"""ネットワークを使う ISMCTS による自己対戦で, 強化学習のデータを生成する.

各局面について, 入力 (planes, scalars), 合法手のマスク, 探索の訪問回数の分布 π (policy), 探索によるルートの価値 q,
最終的な勝敗 z (values, その局面の手番のプレイヤーから見た値) を記録する (選択の余地が無い局面は記録しない).

探索の多様性のため:
    - ルートの事前確率にディリクレノイズを混ぜる (dirichlet_alpha, dirichlet_epsilon)
    - 最初の temperature_moves 手は, 訪問回数に比例した確率で手を選ぶ (それ以降は訪問回数が最大の手)

    python -m az.selfplay --checkpoint checkpoints/pretrain.pt --games 500 --out data/selfplay/gen_0000

CPU アフィニティは match.py と同じく, 環境変数 BL_AFFINITY (または --affinity) で指定できる.
推論は各プロセスの CPU で行う (1プロセス1スレッド).
"""
import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import pybl
from az import features as F
from az.train import concat

ROOT = Path(__file__).resolve().parent.parent      # bl_ex のディレクトリ

DEFAULT_CONFIG = {
    "num_simulations": 200,
    "c_puct": 1.5,
    "batch_size": 16,
    "fpu": 0.0,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,
    "temperature_moves": 16,
}

_evaluators = {}


def _evaluator(checkpoint: str):
    """チェックポイントごとに1回だけ読み込む (ファイルが更新されたら読み込み直す)"""
    key = (checkpoint, os.path.getmtime(checkpoint))
    if key not in _evaluators:
        import torch
        from az.mcts import NetworkEvaluator
        from az.model import load_checkpoint
        torch.set_num_threads(1)
        _evaluators.clear()
        _evaluators[key] = NetworkEvaluator(load_checkpoint(checkpoint, device="cpu")[0], "cpu")
    return _evaluators[key]


def play_selfplay_game(task: tuple) -> dict:
    """自己対戦を1局行い, 記録した局面を配列で返す"""
    from az import mcts
    game_index, checkpoint, seed, config = task
    evaluator = _evaluator(checkpoint)
    rng = random.Random(seed)
    state = pybl.GameState(seed=rng.getrandbits(64))

    planes, scalars, masks, policies, qs, players = [], [], [], [], [], []
    decisions = 0
    while not state.is_terminal:
        player = state.side_to_move
        obs = state.observe(player)
        legal = obs.get_legal_moves()
        if len(legal) == 1:
            state.play(legal[0])
            continue
        result = mcts.search(obs, evaluator, num_simulations=config["num_simulations"], c_puct=config["c_puct"],
                             batch_size=config["batch_size"], fpu=config["fpu"],
                             dirichlet_alpha=config["dirichlet_alpha"],
                             dirichlet_epsilon=config["dirichlet_epsilon"], rng=rng)
        p, s = F.encode(obs)
        planes.append(p.astype(np.uint8))
        scalars.append(s)
        masks.append(F.legal_mask(obs))
        policies.append(result.policy.astype(np.float16))
        qs.append(result.root_value)
        players.append(int(player))
        temperature = 1.0 if decisions < config["temperature_moves"] else 0.0
        state.play(result.sample_move(temperature, rng))
        decisions += 1

    winner = state.winner
    values = [0.0 if winner == pybl.NULL_PLAYER else (1.0 if winner == p else -1.0) for p in players]
    n = len(players)
    return {
        "planes": np.array(planes, dtype=np.uint8).reshape(n, F.NUM_PLANES, *F.PLANE_SHAPE),
        "scalars": np.array(scalars, dtype=np.float32).reshape(n, F.NUM_SCALARS),
        "masks": np.array(masks, dtype=bool).reshape(n, F.NUM_ACTIONS),
        "policy": np.array(policies, dtype=np.float16).reshape(n, F.NUM_ACTIONS),
        "q": np.array(qs, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "games": np.full(n, game_index, dtype=np.int32),
        "winner": np.full(n, int(winner), dtype=np.int8),
    }


def generate(checkpoint, games: int, out, config: dict | None = None, jobs: int = 0, seed=None,
             affinity: str | None = None, save_every: int = 100, log=print) -> dict:
    """自己対戦を games 局行い, out のディレクトリに保存する. 統計 (局数・局面数・時間) を返す"""
    config = {**DEFAULT_CONFIG, **(config or {})}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = str(Path(checkpoint).resolve())
    master = random.Random(seed)
    tasks = [(i, checkpoint, master.getrandbits(64), config) for i in range(games)]

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from match import apply_cpu_affinity
    cpus = apply_cpu_affinity(affinity)
    jobs = max(1, min(games, jobs or (len(cpus) if cpus else os.cpu_count()) or 1))
    log(f"自己対戦 {games} 局 ({jobs} 並列, {config['num_simulations']} シミュレーション/手) → {out}")

    start = time.perf_counter()
    buffer, part, positions = [], 0, 0
    executor = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        results = executor.map(play_selfplay_game, tasks, chunksize=1) if executor else map(play_selfplay_game, tasks)
        for done, record in enumerate(results, 1):
            buffer.append(record)
            positions += len(record["values"])
            if len(buffer) >= save_every or done == games:
                np.savez_compressed(out / f"part_{part:04d}.npz", **concat(buffer))
                buffer, part = [], part + 1
                elapsed = time.perf_counter() - start
                log(f"  {done}/{games} 局 ({positions} 局面) [経過 {elapsed:.0f} 秒, "
                    f"残り約 {elapsed / done * (games - done):.0f} 秒]")
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)
    return {"games": games, "positions": positions, "seconds": time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(description="ネットワークを使う ISMCTS による自己対戦でデータを生成する")
    parser.add_argument("--checkpoint", required=True, help="自己対戦に使うネットワーク")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--out", required=True, help="保存先のディレクトリ")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=0, help="並列化数 (0 で使える論理プロセッサの数)")
    parser.add_argument("--affinity", default=os.environ.get("BL_AFFINITY"),
                        help="使う論理プロセッサの16進数のマスク (省略時は環境変数 BL_AFFINITY)")
    for key, value in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args()
    config = {key: getattr(args, key) for key in DEFAULT_CONFIG}
    generate(args.checkpoint, args.games, args.out, config, jobs=args.jobs, seed=args.seed, affinity=args.affinity,
             log=lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
