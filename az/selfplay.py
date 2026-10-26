"""ネットワークを使う ISMCTS による自己対戦で, 強化学習のデータを生成する.

各局面について, 入力 (planes, scalars), 合法手のマスク, 探索の訪問回数の分布 π (policy), 探索によるルートの価値 q,
最終的な勝敗 z (values, その局面の手番のプレイヤーから見た値) を記録する (選択の余地が無い局面は記録しない).

探索の多様性のため:
    - ルートの事前確率にディリクレノイズを混ぜる (dirichlet_alpha, dirichlet_epsilon)
    - 最初の temperature_moves 手は, 訪問回数に比例した確率で手を選ぶ (それ以降は訪問回数が最大の手)

    python -m az.selfplay --checkpoint checkpoints/pretrain.pt --games 500 --out data/selfplay/gen_0000
    python -m az.selfplay --checkpoint checkpoints/pretrain.pt --games 500 --out data/selfplay/gen_0000 --device cuda --concurrency 32

CPU アフィニティは match.py と同じく, 環境変数 BL_AFFINITY (または --affinity) で指定できる.

並列化: jobs 個のプロセスがそれぞれ concurrency 局を同時に進め, 全ての局の探索の葉をまとめてネットワークで評価する.
GPU があれば推論は GPU (半精度) で行い, 木の操作 (determinization・選択・逆伝播) は各プロセスの CPU で行う.
"""
import argparse
import multiprocessing as mp
import os
import queue
import random
import sys
import time
import traceback
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
    "fpu_reduction": None,          # 指定すると未訪問の手の Q を 親の価値 − fpu_reduction × √(訪問済みの P) にする (fpu は使わない)
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,
    "temperature_moves": 16,
}

CONFIG_HELP = {
    "num_simulations": "1手あたりのシミュレーションの回数",
    "c_puct": "PUCT の探索係数",
    "batch_size": "1局の探索でまとめて評価する葉の数",
    "fpu": "未訪問の手の Q の値",
    "fpu_reduction": "FPU reduction (例: 0.2). 省略時は使わず, fpu を使う",
    "dirichlet_alpha": "ルートに混ぜるディリクレノイズの集中度",
    "dirichlet_epsilon": "ディリクレノイズを混ぜる割合",
    "temperature_moves": "訪問回数に比例した確率で手を選ぶ手数",
}


def add_config_arguments(parser) -> None:
    """自己対戦の設定 (DEFAULT_CONFIG) のコマンドライン引数を加える"""
    for key, value in DEFAULT_CONFIG.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=float if value is None else type(value), default=value,
                            help=f"自己対戦の設定: {CONFIG_HELP[key]}")

def default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_evaluator(checkpoint: str, device: str = "cpu"):
    """チェックポイントを読み込み, ネットワークの評価器を作る (CPU では1スレッドで推論する)"""
    import torch
    from az.mcts import NetworkEvaluator
    from az.model import load_checkpoint
    torch.set_num_threads(1)
    return NetworkEvaluator(load_checkpoint(checkpoint, device=device)[0], device)


def selfplay_game_steps(game_index: int, seed: int, config: dict):
    """自己対戦を1局行うジェネレータ. 探索の評価の要求を yield し (mcts.search_steps と同じ形), 記録した局面を返す"""
    from az import mcts
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
        result = yield from mcts.search_steps(
            obs, num_simulations=config["num_simulations"], c_puct=config["c_puct"],
            batch_size=config["batch_size"], fpu=config["fpu"], fpu_reduction=config.get("fpu_reduction"),
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


def play_selfplay_game(task: tuple, evaluator=None) -> dict:
    """自己対戦を1局行い, 記録した局面を返す. task = (局の番号, チェックポイント, シード, 設定)"""
    from az.mcts import run_steps
    game_index, checkpoint, seed, config = task
    return run_steps(selfplay_game_steps(game_index, seed, config), evaluator or load_evaluator(checkpoint))


def play_games_batched(tasks, evaluator, concurrency: int):
    """tasks の (局の番号, シード, 設定) の自己対戦を最大 concurrency 局ずつ同時に進め, 終わった局の記録を順に yield する.

    各局の探索の評価の要求をつなげて, evaluator で1回にまとめて評価する.
    """
    tasks = iter(tasks)
    active = []         # [局のジェネレータ, 評価の要求]
    exhausted = False
    while True:
        while not exhausted and len(active) < concurrency:
            task = next(tasks, None)
            if task is None:
                exhausted = True
                break
            game = selfplay_game_steps(*task)
            try:
                active.append([game, next(game)])
            except StopIteration as stop:       # 評価が1度も必要ない局
                yield stop.value
        if not active:
            return

        sizes = [len(request[0]) for _, request in active]
        logits, values = evaluator(np.concatenate([request[0] for _, request in active]),
                                   np.concatenate([request[1] for _, request in active]))
        still, offset = [], 0
        for (game, _), size in zip(active, sizes):
            try:
                still.append([game, game.send((logits[offset:offset + size], values[offset:offset + size]))])
            except StopIteration as stop:
                yield stop.value
            offset += size
        active = still


def _worker(checkpoint: str, device: str, concurrency: int, task_queue, result_queue) -> None:
    """自己対戦のプロセス: task_queue から (局の番号, シード, 設定) を受け取り (None で終わり), 記録を result_queue に送る"""
    try:
        evaluator = load_evaluator(checkpoint, device)
        for record in play_games_batched(iter(task_queue.get, None), evaluator, concurrency):
            result_queue.put(record)
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))


def _records(tasks: list, checkpoint: str, device: str, jobs: int, concurrency: int):
    """自己対戦の記録を終わった順に yield する (jobs が 1 ならこのプロセスで行う)"""
    if jobs <= 1:
        yield from play_games_batched(tasks, load_evaluator(checkpoint, device), concurrency)
        return
    ctx = mp.get_context("spawn")
    task_queue, result_queue = ctx.Queue(), ctx.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(jobs):
        task_queue.put(None)
    workers = [ctx.Process(target=_worker, args=(checkpoint, device, concurrency, task_queue, result_queue),
                           daemon=True) for _ in range(jobs)]
    for w in workers:
        w.start()
    try:
        for _ in range(len(tasks)):
            while True:
                try:
                    record = result_queue.get(timeout=10)
                    break
                except queue.Empty:
                    if not any(w.is_alive() for w in workers):
                        raise RuntimeError("自己対戦のプロセスが全て終了しました")
            if isinstance(record, tuple):
                raise RuntimeError(f"自己対戦のプロセスでエラーが起きました:\n{record[1]}")
            yield record
    finally:
        for w in workers:
            if w.is_alive():
                w.terminate()
        for w in workers:
            w.join()


def generate(checkpoint, games: int, out, config: dict | None = None, jobs: int = 0, seed=None,
             affinity: str | None = None, save_every: int = 100, device: str | None = None, concurrency: int = 0,
             log=print) -> dict:
    """自己対戦を games 局行い, out のディレクトリに保存する. 統計 (局数・局面数・時間) を返す.

    device: 推論に使う装置 (省略時は GPU があれば cuda)
    concurrency: 1プロセスで同時に進める局の数 (0 なら cuda で 32, cpu で 1)
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = str(Path(checkpoint).resolve())
    device = device or default_device()
    concurrency = concurrency or (32 if device.startswith("cuda") else 1)
    master = random.Random(seed)
    tasks = [(i, master.getrandbits(64), config) for i in range(games)]

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from match import apply_cpu_affinity
    cpus = apply_cpu_affinity(affinity)
    jobs = max(1, min(games, jobs or (len(cpus) if cpus else os.cpu_count()) or 1))
    log(f"自己対戦 {games} 局 ({jobs} プロセス × 同時 {concurrency} 局, 推論 {device}, "
        f"{config['num_simulations']} シミュレーション/手, c_puct {config['c_puct']}, "
        f"fpu_reduction {config['fpu_reduction']}) → {out}")
    if F._fast is None:
        log("  (Cython 版が無いので Python 版で探索する. python setup.py build_ext --inplace でビルドすると速くなる)")

    start = time.perf_counter()
    buffer, part, positions = [], 0, 0
    for done, record in enumerate(_records(tasks, checkpoint, device, jobs, concurrency), 1):
        buffer.append(record)
        positions += len(record["values"])
        if len(buffer) >= save_every or done == games:
            np.savez_compressed(out / f"part_{part:04d}.npz", **concat(buffer))
            buffer, part = [], part + 1
            elapsed = time.perf_counter() - start
            log(f"  {done}/{games} 局 ({positions} 局面) [経過 {elapsed:.0f} 秒, "
                f"残り約 {elapsed / done * (games - done):.0f} 秒]")
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
    parser.add_argument("--device", default=None, help="推論に使う装置 (cuda / cpu, 省略時は GPU があれば cuda)")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="1プロセスで同時に進める局の数 (0 なら cuda で 32, cpu で 1)")
    add_config_arguments(parser)
    args = parser.parse_args()
    config = {key: getattr(args, key) for key in DEFAULT_CONFIG}
    generate(args.checkpoint, args.games, args.out, config, jobs=args.jobs, seed=args.seed, affinity=args.affinity,
             device=args.device, concurrency=args.concurrency, log=lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
