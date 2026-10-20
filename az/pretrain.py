"""既存のエージェントの棋譜による, ネットワークの教師あり学習 (強化学習の前の初期化).

1. 棋譜の生成 (generate): 教師のエージェント (既定は abdeck) 同士で対戦させ, 各局面の
   入力 (planes, scalars), 合法手のマスク, 教師の選んだ手, 最終的な勝敗を記録する.
   多様な局面を集めるため, 確率 explore で実際にはランダムな手を指す (記録する正解は教師の手のまま).

    python -m az.pretrain generate --games 2000 --teacher abdeck -j 8 --out data/pretrain_abdeck

   CPU アフィニティ (使う論理プロセッサ) は match.py と同じく, 環境変数 BL_AFFINITY (または --affinity) で
   指定できる. 並列実行の子プロセスにも引き継がれる (Windows と Linux で有効, macOS では無視).

2. 学習 (train): 記録した棋譜で, 方策 (教師の手) と価値 (最終的な勝敗) を学習する.
   バッチの各要素にランダムな対称変換 (色の入れ替え・フラッグの左右反転) を施してデータを拡張する.

    python -m az.pretrain train --data data/pretrain_abdeck --epochs 30 --out checkpoints/pretrain.pt

いずれも bl_ex のディレクトリで実行する (教師のエージェントを match.py と同じ方法で読み込むため).
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


# ---------------------------------------------------------------------------
# 棋譜の生成
# ---------------------------------------------------------------------------

_teacher_cache = {}


def _teacher(spec: str):
    if spec not in _teacher_cache:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from match import load_agent
        _teacher_cache[spec] = load_agent(spec)
    return _teacher_cache[spec]


def play_teacher_game(task: tuple) -> dict:
    """教師のエージェント同士で1局を行い, 記録した局面を配列で返す"""
    game_index, teacher_spec, seed, explore = task
    teacher = _teacher(teacher_spec)
    rng = random.Random(seed)
    state = pybl.GameState(seed=rng.getrandbits(64))
    agent_rng = random.Random(rng.getrandbits(64))

    planes, scalars, masks, actions, players = [], [], [], [], []
    while not state.is_terminal:
        player = state.side_to_move
        obs = state.observe(player)
        legal = obs.get_legal_moves()
        if len(legal) == 1:
            state.play(legal[0])        # 選択の余地が無い局面は記録しない
            continue
        move = int(teacher.choose(obs, agent_rng, state))
        p, s = F.encode(obs)
        planes.append(p.astype(np.uint8))
        scalars.append(s)
        masks.append(F.legal_mask(obs))
        actions.append(move)
        players.append(int(player))
        state.play(legal[rng.randrange(len(legal))] if rng.random() < explore else move)

    winner = state.winner
    values = [0.0 if winner == pybl.NULL_PLAYER else (1.0 if winner == p else -1.0) for p in players]
    n = len(actions)
    return {
        "planes": np.array(planes, dtype=np.uint8).reshape(n, F.NUM_PLANES, *F.PLANE_SHAPE),
        "scalars": np.array(scalars, dtype=np.float32).reshape(n, F.NUM_SCALARS),
        "masks": np.array(masks, dtype=bool).reshape(n, F.NUM_ACTIONS),
        "actions": np.array(actions, dtype=np.int16),
        "values": np.array(values, dtype=np.float32),
        "games": np.full(n, game_index, dtype=np.int32),
    }


def generate(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    master = random.Random(args.seed)
    tasks = [(i, args.teacher, master.getrandbits(64), args.explore) for i in range(args.games)]

    # 並列実行のプロセスを作る前に CPU アフィニティを設定する (子プロセスに引き継がれる)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from match import apply_cpu_affinity
    cpus = apply_cpu_affinity(args.affinity)
    if cpus is not None:
        print(f"CPU アフィニティ: 論理プロセッサ {', '.join(map(str, cpus))} を使用", flush=True)
    jobs = max(1, min(args.games, args.jobs or (len(cpus) if cpus else os.cpu_count()) or 1))
    print(f"教師 {args.teacher} の棋譜を {args.games} 局生成します ({jobs} 並列, explore={args.explore}) → {out}",
          flush=True)

    start = time.perf_counter()
    buffer, part, total = [], 0, 0
    executor = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        results = executor.map(play_teacher_game, tasks, chunksize=1) if executor else map(play_teacher_game, tasks)
        for done, record in enumerate(results, 1):
            buffer.append(record)
            total += len(record["actions"])
            if len(buffer) >= args.save_every or done == args.games:
                path = out / f"part_{part:04d}.npz"
                np.savez_compressed(path, **concat(buffer))
                buffer, part = [], part + 1
                elapsed = time.perf_counter() - start
                print(f"  {done}/{args.games} 局 ({total} 局面) を保存 [経過 {elapsed:.0f} 秒, "
                      f"残り約 {elapsed / done * (args.games - done):.0f} 秒]", flush=True)
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------

def train(args) -> None:
    from az.model import build_model
    from az.train import fit, load_dataset

    if args.arch == "token":
        config = {"arch": "token", "d_model": args.d_model, "layers": args.layers, "heads": args.heads}
    else:
        config = {"arch": "mlp", "hidden": args.hidden, "blocks": args.blocks}
    model = build_model(config)
    print(f"ネットワーク: {model.config}, パラメータ数 {sum(p.numel() for p in model.parameters()):,}, "
          f"価値の損失の重み {args.value_weight}", flush=True)
    fit(model, load_dataset(args.data), out=args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, value_weight=args.value_weight, augment=not args.no_augment,
        val_ratio=args.val_ratio, device=args.device, seed=args.seed, keep_best=True,
        log=lambda text: print(text, flush=True), kind="pretrain", data=args.data)
    print(f"検証の損失が最も小さかったネットワークを {args.out} に保存しました")


def main():
    parser = argparse.ArgumentParser(description="既存のエージェントの棋譜による教師あり学習")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="教師のエージェントの棋譜を生成する")
    g.add_argument("--games", type=int, default=2000, help="生成する局数")
    g.add_argument("--teacher", default="abdeck", help="教師のエージェント (match.py と同じ指定方法)")
    g.add_argument("--explore", type=float, default=0.1, help="実際にはランダムな手を指す確率 (局面の多様化)")
    g.add_argument("--out", default="data/pretrain_abdeck", help="保存先のディレクトリ")
    g.add_argument("--save-every", type=int, default=200, help="何局ごとにファイルに保存するか")
    g.add_argument("--seed", type=int, default=None, help="乱数シード")
    g.add_argument("-j", "--jobs", type=int, default=0, help="並列化数 (0 で使える論理プロセッサの数)")
    g.add_argument("--affinity", default=os.environ.get("BL_AFFINITY"),
                   help="使う論理プロセッサを16進数のマスクで指定する (例: 5555). "
                        "省略時は環境変数 BL_AFFINITY の値. none で指定しない")

    t = sub.add_parser("train", help="棋譜でネットワークを学習する")
    t.add_argument("--data", nargs="+", default=["data/pretrain_abdeck"], help="棋譜のディレクトリ (複数可)")
    t.add_argument("--out", default="checkpoints/pretrain.pt", help="保存するチェックポイント")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3, help="学習率の最大値 (OneCycle)")
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--arch", choices=["token", "mlp"], default="token", help="ネットワークの種類")
    t.add_argument("--d-model", type=int, default=128, help="token: トークンの次元")
    t.add_argument("--layers", type=int, default=4, help="token: Transformer の層の数")
    t.add_argument("--heads", type=int, default=4, help="token: アテンションのヘッドの数")
    t.add_argument("--hidden", type=int, default=512, help="mlp: 隠れ層の次元")
    t.add_argument("--blocks", type=int, default=6, help="mlp: 残差ブロックの数")
    t.add_argument("--value-weight", type=float, default=1.0, help="損失における価値の二乗誤差の重み")
    t.add_argument("--val-ratio", type=float, default=0.05, help="検証用に使う局の割合")
    t.add_argument("--no-augment", action="store_true", help="対称変換によるデータ拡張をしない")
    t.add_argument("--device", default=None, help="cuda / cpu (省略時は GPU があれば cuda)")
    t.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "generate":
        generate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
