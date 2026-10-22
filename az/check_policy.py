"""自己対戦のデータの方策の正解 (訪問回数の分布 π) が, 世代ごとにどれだけ平らになっているかを調べる.

方策の損失 (交差エントロピー) は H(π) + KL(π‖p) に分けられる. H(π) が世代ごとに増えていれば,
損失の増加はネットワークの合わなさ (KL) ではなく, 正解の分布 π が平らになったことによる.

各世代のデータについて次を表示する (局面ごとの値の平均):
    局面数, 合法手の数, H(π) (π のエントロピー), H(π)/log(合法手の数) (一様分布で 1),
    exp(H(π)) (実質的な候補手の数), π の最大の確率

--checkpoint を指定すると, 同じ局面でのネットワークの方策 p について次も表示する:
    H(p) (事前確率のエントロピー), 交差エントロピー H(π)+KL, KL(π‖p), π と p の最善手の一致率

    python -m az.check_policy --run runs/az3
    python -m az.check_policy --run runs/az3 --checkpoint runs/az3/checkpoints/gen_0000.pt runs/az3/checkpoints/gen_0006.pt
    python -m az.check_policy --data runs/az3/data/gen_0001 runs/az3/data/gen_0007 --max-files 0

bl_ex のディレクトリで実行する.
"""
import argparse
import glob
import os
import random
import unicodedata
from pathlib import Path

import numpy as np

EPS = 1e-12


def _rjust(text: str, width: int) -> str:
    """全角文字を幅 2 として右に揃える (見出しの列を揃えるため)"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return " " * max(0, width - w) + text


def _header(columns: list) -> str:
    return "".join(_rjust(name, width) if i else name + " " * (width - 2 * len(name))
                   for i, (name, width) in enumerate(columns))


def _entropy(p: np.ndarray) -> np.ndarray:
    """各行の分布のエントロピー (自然対数)"""
    return -(p * np.log(np.clip(p, EPS, 1.0))).sum(axis=1)


def policy_stats(policy: np.ndarray, masks: np.ndarray) -> dict:
    """方策の正解 π (N, 541) と合法手のマスク (N, 541) から, π の平らさの統計 (局面ごとの値の平均) を返す"""
    p = policy.astype(np.float32)
    p = p / np.clip(p.sum(axis=1, keepdims=True), EPS, None)       # float16 の丸めによる合計のずれを直す
    legal = masks.sum(axis=1).astype(np.float32)
    h = _entropy(p)
    return {
        "positions": len(p),
        "legal": float(legal.mean()),
        "entropy": float(h.mean()),
        "normalized_entropy": float((h / np.log(np.maximum(legal, 2.0))).mean()),
        "effective_moves": float(np.exp(h).mean()),
        "max_prob": float(p.max(axis=1).mean()),
    }


def network_stats(model, data: dict, device: str = "cpu", batch_size: int = 4096) -> dict:
    """同じ局面でのネットワークの方策 p と π を比べた統計 (局面ごとの値の平均) を返す"""
    import torch
    from az.model import masked_log_softmax

    target = data["policy"].astype(np.float32)
    target = target / np.clip(target.sum(axis=1, keepdims=True), EPS, None)
    totals = {"net_entropy": 0.0, "cross_entropy": 0.0, "kl": 0.0, "top1_agree": 0.0}
    n = len(target)
    model = model.to(device).eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            planes = torch.as_tensor(data["planes"][sl], dtype=torch.float32, device=device)
            scalars = torch.as_tensor(data["scalars"][sl], dtype=torch.float32, device=device)
            masks = torch.as_tensor(data["masks"][sl], dtype=torch.bool, device=device)
            logits, _ = model(planes, scalars)
            log_p = masked_log_softmax(logits, masks).cpu().numpy().astype(np.float64)
            pi = target[sl].astype(np.float64)
            p = np.exp(log_p)
            ce = -(pi * log_p).sum(axis=1)
            totals["net_entropy"] += float(-(p * log_p).sum())
            totals["cross_entropy"] += float(ce.sum())
            totals["kl"] += float((ce - _entropy(pi)).sum())
            totals["top1_agree"] += float((p.argmax(axis=1) == pi.argmax(axis=1)).sum())
    return {key: value / n for key, value in totals.items()}


def load_sample(directory, max_files: int, seed: int = 0) -> dict | None:
    """ディレクトリの npz ファイルから, 最大 max_files 個 (0 なら全て) を選んで読み込む. 方策の分布が無ければ None"""
    files = sorted(glob.glob(os.path.join(str(directory), "*.npz")))
    if max_files > 0 and len(files) > max_files:
        files = sorted(random.Random(seed).sample(files, max_files))
    keys = ("planes", "scalars", "masks", "policy")
    parts = []
    for path in files:
        with np.load(path) as z:
            if "policy" not in z.files:
                return None
            parts.append({k: z[k] for k in keys})
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="自己対戦のデータの方策の正解 π の平らさを世代ごとに調べる")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="az.loop の実行のディレクトリ (data/gen_* を全て調べる)")
    source.add_argument("--data", nargs="+", help="調べるデータのディレクトリ (複数可)")
    parser.add_argument("--checkpoint", nargs="*", default=[], help="同じ局面で方策を比べるネットワーク (複数可)")
    parser.add_argument("--max-files", type=int, default=10, help="各世代で読み込むファイルの数 (0 で全て)")
    parser.add_argument("--device", default=None, help="ネットワークの推論に使う装置 (省略時は GPU があれば cuda)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    directories = sorted(Path(args.run).glob("data/gen_*")) if args.run else [Path(d) for d in args.data]
    if not directories:
        raise SystemExit("データのディレクトリが見つかりません")

    models = []
    if args.checkpoint:
        from az.model import default_device, load_checkpoint
        device = args.device or str(default_device())
        models = [(Path(c).stem, load_checkpoint(c, device=device)[0]) for c in args.checkpoint]

    print(_header([("データ", 12), ("局面数", 9), ("合法手", 8), ("H(π)", 8), ("H/logN", 8), ("exp(H)", 8),
                   ("最大確率", 10)]))
    rows = []
    for directory in directories:
        data = load_sample(directory, args.max_files, args.seed)
        if data is None:
            print(f"{directory.name:<12}  (方策の分布 π が無いので飛ばす)")
            continue
        s = policy_stats(data["policy"], data["masks"])
        print(f"{directory.name:<12}{s['positions']:>9}{s['legal']:>8.1f}{s['entropy']:>8.3f}"
              f"{s['normalized_entropy']:>8.3f}{s['effective_moves']:>8.2f}{s['max_prob']:>9.3f}", flush=True)
        rows.append((directory, data, s))

    for name, model in models:
        print(f"\nネットワーク {name} の方策 p との比較")
        print(_header([("データ", 12), ("H(π)", 8), ("H(p)", 8), ("交差エントロピー", 18), ("KL(π‖p)", 10),
                       ("最善手の一致率", 16)]))
        for directory, data, s in rows:
            n = network_stats(model, data, device=device)
            print(f"{directory.name:<12}{s['entropy']:>8.3f}{n['net_entropy']:>8.3f}{n['cross_entropy']:>18.3f}"
                  f"{n['kl']:>10.3f}{n['top1_agree'] * 100:>15.1f}%", flush=True)


if __name__ == "__main__":
    main()
