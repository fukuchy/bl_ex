"""診断: ISMCTS でルート以外のノードの事前確率 (ネットワークの方策のロジット) を, 評価したときとは別の手札の
determinization でどれだけ使い回しているかを測る.

自己対戦と同じ探索の設定で games 局を対局し, その間の全ての探索について, ルート以外のノードで手を選ぶたびに次を集計する
(深さ 1, 2, 3 以上 × 手番がルートのプレイヤーか相手か):
    手札の不一致: 今の determinization の手番のプレイヤーの手札が, 使うロジットを評価したときの手札と違う割合
    無いカードの手 (単純): 今の合法手のうち, 評価したときの手札に無かったカードを使う手の割合 (選択ごとの平均)
    無いカードの手 (訪問): 同じ割合を, 子の訪問回数で重み付けしたもの
prior_cache=hand では使うロジットは常に今の手札のものなので, 不一致は 0 になる. 代わりに, ノードに来たときに
その手札のロジットがまだ無かった (そこを葉として評価した) 割合と, 葉の深さを見る.

    python -m az.diag_prior --checkpoint runs/az6/checkpoints/gen_0021.pt --games 50 --device cuda
    python -m az.diag_prior --checkpoint runs/az6/checkpoints/gen_0021.pt --games 50 --device cuda --prior-cache hand

探索の設定は az.selfplay と同じ引数で指定する (既定は 200 シミュレーション, c_puct 1.0, fpu_reduction 0.2, batch_size 16.
ディリクレノイズと温度も自己対戦と同じ). bl_ex のディレクトリで実行する.
"""
import argparse
import random
import time

from az import selfplay
from az.check_policy import _header
from az.mcts import PriorStats

DEPTHS = {1: "1", 2: "2", 3: "3+"}


def format_stats(stats: PriorStats) -> str:
    """集計の結果を表にする"""
    leaves = stats.leaves
    count = leaves["eval"] + leaves["terminal"] + leaves["collision"]
    lines = [f"探索 {stats.searches} 回, シミュレーション {count} 回 (葉: ネットワークで評価 {leaves['eval']}, "
             f"終局 {leaves['terminal']}, 評価待ちと重なった {leaves['collision']}), "
             f"葉の深さの平均 {leaves['depth'] / max(count, 1):.2f}", ""]
    lines.append(_header([("深さ", 6), ("手番", 8), ("選択の回数", 12), ("手札の不一致", 14),
                          ("無いカードの手(単純)", 22), ("無いカードの手(訪問)", 22)]))
    for (depth, own), row in sorted(stats.select.items()):
        n, mismatch, simple, weighted, total = row
        lines.append(f"{DEPTHS[depth]:<6}{'自分' if own else '相手':>6}{n:>12}{mismatch / n * 100:>13.1f}%"
                     f"{simple / n * 100:>21.1f}%{(weighted / total * 100 if total else 0.0):>21.1f}%")
    if stats.arrive:
        lines += ["", "prior_cache=hand: ノードに来たときに, その手札のロジットがまだ無かった割合"]
        lines.append(_header([("深さ", 6), ("手番", 8), ("到達の回数", 12), ("まだ無かった", 14)]))
        for (depth, own), (n, miss) in sorted(stats.arrive.items()):
            lines.append(f"{DEPTHS[depth]:<6}{'自分' if own else '相手':>6}{n:>12}{miss / n * 100:>13.1f}%")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ISMCTS の事前確率の使い回しの度合いを, 自己対戦の探索で測る")
    parser.add_argument("--checkpoint", required=True, help="使うネットワーク")
    parser.add_argument("--games", type=int, default=50, help="対局の数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="推論に使う装置 (cuda / cpu, 省略時は GPU があれば cuda)")
    parser.add_argument("--concurrency", type=int, default=0, help="同時に進める局の数 (0 なら cuda で 32, cpu で 1)")
    selfplay.add_config_arguments(parser)
    parser.set_defaults(c_puct=1.0, fpu_reduction=0.2)
    args = parser.parse_args()

    config = {key: getattr(args, key) for key in selfplay.DEFAULT_CONFIG}
    device = args.device or selfplay.default_device()
    concurrency = args.concurrency or (32 if device.startswith("cuda") else 1)
    evaluator = selfplay.load_evaluator(args.checkpoint, device)
    stats = PriorStats()
    master = random.Random(args.seed)
    tasks = [(i, master.getrandbits(64), config, stats) for i in range(args.games)]
    print(f"{args.games} 局 ({config['num_simulations']} シミュレーション/手, c_puct {config['c_puct']}, "
          f"fpu_reduction {config['fpu_reduction']}, batch_size {config['batch_size']}, "
          f"prior_cache {config['prior_cache']}, 推論 {device})", flush=True)
    start = time.perf_counter()
    for done, _ in enumerate(selfplay.play_games_batched(tasks, evaluator, concurrency), 1):
        if done % 10 == 0 or done == args.games:
            print(f"  {done}/{args.games} 局 [経過 {time.perf_counter() - start:.0f} 秒]", flush=True)
    print()
    print(format_stats(stats))


if __name__ == "__main__":
    main()
