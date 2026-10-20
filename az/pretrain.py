"""既存のエージェントの棋譜による, ネットワークの教師あり学習 (強化学習の前の初期化).

1. 棋譜の生成 (generate): 教師のエージェント (既定は abdeck) 同士で対戦させ, 各局面の
   入力 (planes, scalars), 合法手のマスク, 教師の選んだ手, 最終的な勝敗を記録する.
   多様な局面を集めるため, 確率 explore で実際にはランダムな手を指す (記録する正解は教師の手のまま).

    python -m az.pretrain generate --games 2000 --teacher abdeck -j 8 --out data/pretrain_abdeck

2. 学習 (train): 記録した棋譜で, 方策 (教師の手) と価値 (最終的な勝敗) を学習する.
   バッチの各要素にランダムな対称変換 (色の入れ替え・フラッグの左右反転) を施してデータを拡張する.

    python -m az.pretrain train --data data/pretrain_abdeck --epochs 30 --out checkpoints/pretrain.pt

いずれも bl_ex のディレクトリで実行する (教師のエージェントを match.py と同じ方法で読み込むため).
"""
import argparse
import glob
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import pybl
from az import features as F

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


def _concat(parts: list) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def generate(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    master = random.Random(args.seed)
    tasks = [(i, args.teacher, master.getrandbits(64), args.explore) for i in range(args.games)]
    jobs = max(1, min(args.games, args.jobs or os.cpu_count() or 1))
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
                np.savez_compressed(path, **_concat(buffer))
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

def load_dataset(directories: list) -> dict:
    files = sorted(f for d in directories for f in glob.glob(os.path.join(d, "*.npz")))
    if not files:
        raise SystemExit(f"棋譜が見つかりません: {', '.join(directories)}")
    parts = []
    offset = 0
    for path in files:
        with np.load(path) as data:
            part = {k: data[k] for k in data.files}
        part["games"] = part["games"] + offset         # 複数のディレクトリで局の番号が重ならないようにする
        offset = int(part["games"].max()) + 1
        parts.append(part)
    return _concat(parts)


def train(args) -> None:
    import torch
    from az.model import SymmetryAugmenter, build_model, default_device, loss_fn, masked_log_softmax, save_checkpoint

    device = torch.device(args.device) if args.device else default_device()
    torch.manual_seed(args.seed)
    data = load_dataset(args.data)
    n = len(data["actions"])

    # 局単位で学習用と検証用に分ける (同じ局の局面が両方に入らないようにする)
    games = np.unique(data["games"])
    rng = np.random.default_rng(args.seed)
    val_games = set(rng.choice(games, size=max(1, int(len(games) * args.val_ratio)), replace=False).tolist())
    is_val = np.array([g in val_games for g in data["games"]])
    print(f"局面数 {n} (学習 {int((~is_val).sum())}, 検証 {int(is_val.sum())}), 局数 {len(games)}, 装置 {device}",
          flush=True)

    def to_device(idx):
        return {
            "planes": torch.as_tensor(data["planes"][idx], device=device),            # uint8 のまま置く
            "scalars": torch.as_tensor(data["scalars"][idx], device=device),
            "masks": torch.as_tensor(data["masks"][idx], device=device),
            "actions": torch.as_tensor(data["actions"][idx].astype(np.int64), device=device),
            "values": torch.as_tensor(data["values"][idx], device=device),
        }

    train_set, val_set = to_device(np.flatnonzero(~is_val)), to_device(np.flatnonzero(is_val))
    if args.arch == "token":
        config = {"arch": "token", "d_model": args.d_model, "layers": args.layers, "heads": args.heads}
    else:
        config = {"arch": "mlp", "hidden": args.hidden, "blocks": args.blocks}
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_set["actions"]) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    augment = SymmetryAugmenter(device) if not args.no_augment else None
    print(f"ネットワーク: {model.config}, パラメータ数 {sum(p.numel() for p in model.parameters()):,}, "
          f"価値の損失の重み {args.value_weight}", flush=True)

    def batch_tensors(subset, idx):
        planes = subset["planes"][idx].float()
        scalars = subset["scalars"][idx]
        masks = subset["masks"][idx]
        policy = torch.zeros(len(idx), F.NUM_ACTIONS, device=device)
        policy.scatter_(1, subset["actions"][idx][:, None], 1.0)
        return planes, scalars, policy, masks, subset["values"][idx]

    @torch.no_grad()
    def evaluate(subset):
        model.eval()
        total = {"loss": 0.0, "value": 0.0, "policy": 0.0, "acc": 0.0}
        count = len(subset["actions"])
        for start in range(0, count, 4096):
            idx = torch.arange(start, min(start + 4096, count), device=device)
            planes, scalars, policy, masks, values = batch_tensors(subset, idx)
            logits, value = model(planes, scalars)
            loss, vl, pl = loss_fn(logits, value, policy, values, masks, args.value_weight)
            k = len(idx)
            total["loss"] += loss.item() * k
            total["value"] += vl.item() * k
            total["policy"] += pl.item() * k
            total["acc"] += (masked_log_softmax(logits, masks).argmax(1) == subset["actions"][idx]).float().sum().item()
        model.train()
        return {key: v / count for key, v in total.items()}

    best = math.inf
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(train_set["actions"]), device=device)
        running = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            planes, scalars, policy, masks, values = batch_tensors(train_set, idx)
            if augment:
                planes, scalars, policy, masks = augment(planes, scalars, policy, masks)
            logits, value = model(planes, scalars)
            loss, _, _ = loss_fn(logits, value, policy, values, masks, args.value_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()

        val = evaluate(val_set)
        mark = ""
        if val["loss"] < best:
            best = val["loss"]
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.out, model, epoch=epoch, val=val, data=args.data, kind="pretrain")
            mark = "  (保存)"
        print(f"epoch {epoch:3d}  学習の損失 {running / steps_per_epoch:.4f}  |  検証: 損失 {val['loss']:.4f} "
              f"(価値 {val['value']:.4f}, 方策 {val['policy']:.4f}), 教師の手との一致率 {val['acc'] * 100:.1f}%"
              f"  [{time.perf_counter() - start_time:.0f} 秒]{mark}", flush=True)
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
    g.add_argument("-j", "--jobs", type=int, default=0, help="並列化数 (0 で CPU のコア数)")

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
