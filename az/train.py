"""ネットワークの学習 (教師あり学習と強化学習で共通の処理) と, 自己対戦のデータによる学習のコマンド.

データ (npz ファイルを並べたディレクトリ) の各配列:
    planes (N, 20, 6, 10) uint8, scalars (N, 41) float32, masks (N, 541) bool, values (N,) float32 (最終的な勝敗 z),
    games (N,) int32 (局の番号), および方策の正解として次のどちらか
        actions (N,) int16       : 教師の選んだ手 (教師あり学習)
        policy  (N, 541) float16 : 探索の訪問回数の分布 π (自己対戦)
    任意で q (N,) float32       : 探索によるルートの価値 (価値の正解に混ぜられる)

自己対戦のデータによる学習 (az/loop.py からも呼ばれる):
    python -m az.train --data runs/az1/data/gen_0001 runs/az1/data/gen_0002 --init runs/az1/checkpoints/gen_0002.pt \\
        --out runs/az1/checkpoints/gen_0003.pt --epochs 2
"""
import argparse
import glob
import math
import os
import time
from pathlib import Path

import numpy as np

from az import features as F


def concat(parts: list) -> dict:
    keys = set(parts[0])
    for p in parts[1:]:
        keys &= set(p)
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def load_dataset(directories: list) -> dict:
    """ディレクトリ内の npz ファイルを全て読み込んでつなげる (局の番号はディレクトリをまたいで重ならないようにする)"""
    files = sorted(f for d in directories for f in glob.glob(os.path.join(str(d), "*.npz")))
    if not files:
        raise SystemExit(f"データが見つかりません: {', '.join(map(str, directories))}")
    parts, offset = [], 0
    for path in files:
        with np.load(path) as data:
            part = {k: data[k] for k in data.files}
        part["games"] = part["games"] + offset
        offset = int(part["games"].max()) + 1
        parts.append(part)
    return concat(parts)


def fit(model, dataset: dict, *, out, epochs: int = 1, batch_size: int = 1024, lr: float = 1e-3,
        weight_decay: float = 1e-4, value_weight: float = 1.0, q_weight: float = 0.0, augment: bool = True,
        val_ratio: float = 0.05, device=None, seed: int = 0, keep_best: bool = True, log=print, **extra) -> dict:
    """model を dataset で学習し, out に保存する. 最後のエポックの検証の結果を返す.

    value_weight: 損失における価値の二乗誤差の重み
    q_weight: 価値の正解を (1 - q_weight) × z + q_weight × q にする (データに q がある場合のみ)
    keep_best: True なら検証の損失が最も小さいエポックを保存し, False なら最後のエポックを保存する
    extra: チェックポイントに一緒に保存する情報
    """
    import torch
    from az.model import SymmetryAugmenter, default_device, loss_fn, masked_log_softmax, save_checkpoint

    device = torch.device(device) if device else default_device()
    torch.manual_seed(seed)
    model = model.to(device).train()
    data = dataset
    soft = "policy" in data
    n = len(data["values"])
    values = data["values"].astype(np.float32)
    if q_weight > 0 and "q" in data:
        values = (1 - q_weight) * values + q_weight * data["q"].astype(np.float32)

    # 局単位で学習用と検証用に分ける (同じ局の局面が両方に入らないようにする)
    games = np.unique(data["games"])
    rng = np.random.default_rng(seed)
    num_val = int(len(games) * val_ratio)
    val_games = set(rng.choice(games, size=num_val, replace=False).tolist()) if num_val > 0 else set()
    is_val = np.isin(data["games"], list(val_games))
    log(f"局面数 {n} (学習 {int((~is_val).sum())}, 検証 {int(is_val.sum())}), 局数 {len(games)}, 装置 {device}, "
        f"方策の正解 {'訪問回数の分布' if soft else '教師の手'}")

    def to_device(idx):
        subset = {
            "planes": torch.as_tensor(data["planes"][idx], device=device),            # uint8 のまま置く
            "scalars": torch.as_tensor(data["scalars"][idx], device=device),
            "masks": torch.as_tensor(data["masks"][idx], device=device),
            "values": torch.as_tensor(values[idx], device=device),
        }
        if soft:
            subset["policy"] = torch.as_tensor(data["policy"][idx], device=device)     # float16 のまま置く
        else:
            subset["actions"] = torch.as_tensor(data["actions"][idx].astype(np.int64), device=device)
        return subset

    train_set, val_set = to_device(np.flatnonzero(~is_val)), to_device(np.flatnonzero(is_val))
    num_train = len(train_set["values"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, math.ceil(num_train / batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch)
    augmenter = SymmetryAugmenter(device) if augment else None

    def batch_tensors(subset, idx):
        planes = subset["planes"][idx].float()
        if soft:
            policy = subset["policy"][idx].float()
        else:
            policy = torch.zeros(len(idx), F.NUM_ACTIONS, device=device)
            policy.scatter_(1, subset["actions"][idx][:, None], 1.0)
        return planes, subset["scalars"][idx], policy, subset["masks"][idx], subset["values"][idx]

    @torch.no_grad()
    def evaluate(subset):
        count = len(subset["values"])
        if count == 0:
            return None
        model.eval()
        total = {"loss": 0.0, "value": 0.0, "policy": 0.0, "acc": 0.0}
        for start in range(0, count, 4096):
            idx = torch.arange(start, min(start + 4096, count), device=device)
            planes, scalars, policy, masks, target_values = batch_tensors(subset, idx)
            logits, value = model(planes, scalars)
            loss, vl, pl = loss_fn(logits, value, policy, target_values, masks, value_weight)
            k = len(idx)
            total["loss"] += loss.item() * k
            total["value"] += vl.item() * k
            total["policy"] += pl.item() * k
            total["acc"] += (masked_log_softmax(logits, masks).argmax(1) == policy.argmax(1)).float().sum().item()
        model.train()
        return {key: v / count for key, v in total.items()}

    best = math.inf
    val = None
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_train, device=device)
        running = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            planes, scalars, policy, masks, target_values = batch_tensors(train_set, idx)
            if augmenter:
                planes, scalars, policy, masks = augmenter(planes, scalars, policy, masks)
            logits, value = model(planes, scalars)
            loss, _, _ = loss_fn(logits, value, policy, target_values, masks, value_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()

        val = evaluate(val_set)
        mark = ""
        # keep_best なら検証の損失が最小のエポックを, そうでなければ (または検証データが無ければ) 毎エポック保存する
        if not (keep_best and val) or val["loss"] < best:
            if val:
                best = min(best, val["loss"])
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(out, model, epoch=epoch, val=val, **extra)
            mark = "  (保存)"
        val_text = (f"検証: 損失 {val['loss']:.4f} (価値 {val['value']:.4f}, 方策 {val['policy']:.4f}), "
                    f"正解の手との一致率 {val['acc'] * 100:.1f}%") if val else "検証データなし"
        log(f"epoch {epoch:3d}  学習の損失 {running / steps_per_epoch:.4f}  |  {val_text}"
            f"  [{time.perf_counter() - start_time:.0f} 秒]{mark}")
    model.eval()
    return {"train_loss": running / steps_per_epoch, "val": val}


def main():
    parser = argparse.ArgumentParser(description="自己対戦のデータでネットワークを学習する")
    parser.add_argument("--data", nargs="+", required=True, help="データのディレクトリ (複数可)")
    parser.add_argument("--init", required=True, help="学習を始めるネットワーク (チェックポイント)")
    parser.add_argument("--out", required=True, help="保存するチェックポイント")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=0.0, help="価値の正解に混ぜる探索の価値 q の割合")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from az.model import load_checkpoint
    model, _ = load_checkpoint(args.init, device="cpu")
    fit(model, load_dataset(args.data), out=args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, value_weight=args.value_weight, q_weight=args.q_weight,
        augment=not args.no_augment, val_ratio=args.val_ratio, device=args.device, seed=args.seed,
        keep_best=False, kind="selfplay", data=[str(d) for d in args.data], init=str(args.init))


if __name__ == "__main__":
    main()
