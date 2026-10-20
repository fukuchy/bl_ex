"""2つのエージェントを対戦させて勝率を求める (強化学習のゲーティングと, 既存のエージェントとの定期的な比較).

match.py の対戦の仕組み (GameTask, run_game) をそのまま使う. エージェントは match.py と同じ方法で指定する.

    python -m az.evaluate "az@checkpoint=runs/az1/checkpoints/gen_0003.pt" abdeck --games 200
"""
import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # bl_ex のディレクトリ


def _wilson(score: float, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return 0.0, 1.0
    p = score / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def evaluate(agent1: str, agent2: str, games: int, jobs: int = 0, seed=None, affinity: str | None = None) -> dict:
    """agent1 と agent2 を games 試合 (先後交互) 対戦させ, agent1 から見た成績を返す"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import match

    cpus = match.apply_cpu_affinity(affinity)
    jobs = max(1, min(games, jobs or (len(cpus) if cpus else os.cpu_count()) or 1))
    master = random.Random(seed)
    tasks = []
    for game in range(games):
        state_seed, first_seed, second_seed = master.getrandbits(64), master.getrandbits(64), master.getrandbits(64)
        tasks.append(match.GameTask(game, agent1, agent2, game % 2 == 0, state_seed, first_seed, second_seed))

    start = time.perf_counter()
    wins = losses = draws = 0
    executor = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        results = executor.map(match.run_game, tasks, chunksize=1) if executor else map(match.run_game, tasks)
        for result in results:
            if result.winner == 1:
                wins += 1
            elif result.winner == 2:
                losses += 1
            else:
                draws += 1
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)

    score = wins + 0.5 * draws
    low, high = _wilson(score, games)
    return {"wins": wins, "losses": losses, "draws": draws, "games": games, "rate": score / games,
            "ci_low": low, "ci_high": high, "seconds": time.perf_counter() - start}


def format_result(result: dict) -> str:
    return (f"{result['wins']}勝 {result['losses']}敗 {result['draws']}分 (勝率 {result['rate'] * 100:.1f}%, "
            f"95%区間 {result['ci_low'] * 100:.1f}〜{result['ci_high'] * 100:.1f}%, {result['seconds']:.0f} 秒)")


def main():
    parser = argparse.ArgumentParser(description="2つのエージェントを対戦させて勝率を求める")
    parser.add_argument("agent1")
    parser.add_argument("agent2")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-j", "--jobs", type=int, default=0)
    parser.add_argument("--affinity", default=os.environ.get("BL_AFFINITY"))
    args = parser.parse_args()
    result = evaluate(args.agent1, args.agent2, args.games, args.jobs, args.seed, args.affinity)
    print(f"{args.agent1} 対 {args.agent2}: {format_result(result)}")


if __name__ == "__main__":
    main()
