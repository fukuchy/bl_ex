"""事前学習のエポック数の検証: 局数とエポック数を変えて学習し, エポックごとに固定の評価用データでの損失を記録する.

棋譜の一部 (--test-games 局) を評価用として取り分け, どの学習にも使わない. 残りの局から, 各実験で指定した局数を
使って学習する. 各実験は同じ初期値 (--seed) のネットワークから始める. 学習の設定は az.pretrain train と同じ.

    python -m az.epoch_sweep --data data/pretrain_abdeck2 --runs 30000x30 10000x30 3000x30 30000x10 \\
        --d-model 256 --layers 6 --heads 8 --value-weight 0.25 --out runs/epoch_sweep

    --runs の各要素は「局数x エポック数」. 局数に all を書くと, 評価用を除いた全ての局を使う

結果は --out のディレクトリに, 実験ごとの CSV (エポックごとの値) と, 全体の summary.json を保存する.
最後に, 実験ごとに評価用データでの損失が最小になったエポックの一覧を表示する.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from az.train import fit, load_dataset


def parse_run(text: str) -> tuple:
    games, epochs = text.lower().split("x")
    return (None if games == "all" else int(games)), int(epochs)


def split_games(data: dict, test_games: int, seed: int) -> tuple:
    """局の番号で評価用と学習用に分ける. 返り値: (評価用のデータ, 学習用の局の番号 (ランダムな順))"""
    games = np.unique(data["games"])
    rng = np.random.default_rng(seed)
    games = rng.permutation(games)
    if test_games >= len(games):
        raise SystemExit(f"評価用の局数 {test_games} が棋譜の局数 {len(games)} 以上です")
    test_ids, train_ids = games[:test_games], games[test_games:]
    test = {k: v[np.isin(data["games"], test_ids)] for k, v in data.items()}
    return test, train_ids


def subset(data: dict, game_ids) -> dict:
    keep = np.isin(data["games"], game_ids)
    return {k: v[keep] for k, v in data.items()}


def make_evaluator(test: dict, device, value_weight: float, batch: int = 4096):
    """評価用データでの損失 (value_weight × 価値 + 方策) と, 正解の手との一致率を返す関数を作る"""
    import torch
    from az.model import loss_fn, masked_log_softmax
    from az import features as F

    soft = "policy" in test
    n = len(test["values"])
    values = test["values"].astype(np.float32)

    @torch.no_grad()
    def evaluate(model) -> dict:
        was_training = model.training
        model.eval()
        total = {"value": 0.0, "policy": 0.0, "acc": 0.0}
        for start in range(0, n, batch):
            sl = slice(start, min(start + batch, n))
            planes = torch.as_tensor(test["planes"][sl], device=device).float()
            scalars = torch.as_tensor(test["scalars"][sl], device=device)
            masks = torch.as_tensor(test["masks"][sl], device=device)
            target_values = torch.as_tensor(values[sl], device=device)
            if soft:
                policy = torch.as_tensor(test["policy"][sl], device=device).float()
            else:
                actions = torch.as_tensor(test["actions"][sl].astype(np.int64), device=device)
                policy = torch.zeros(len(actions), F.NUM_ACTIONS, device=device)
                policy.scatter_(1, actions[:, None], 1.0)
            logits, value = model(planes, scalars)
            _, vl, pl = loss_fn(logits, value, policy, target_values, masks)
            k = sl.stop - sl.start
            total["value"] += vl.item() * k
            total["policy"] += pl.item() * k
            total["acc"] += (masked_log_softmax(logits, masks).argmax(1) == policy.argmax(1)).float().sum().item()
        if was_training:
            model.train()
        result = {key: v / n for key, v in total.items()}
        result["loss"] = value_weight * result["value"] + result["policy"]
        return result

    return evaluate


def best_epochs(rows: list) -> dict:
    """評価用データでの損失・方策の損失・価値の損失・一致率が最も良かったエポック"""
    return {
        "loss": min(rows, key=lambda r: r["test_loss"])["epoch"],
        "policy": min(rows, key=lambda r: r["test_policy"])["epoch"],
        "value": min(rows, key=lambda r: r["test_value"])["epoch"],
        "acc": max(rows, key=lambda r: r["test_acc"])["epoch"],
    }


def run_one(data: dict, train_ids, test_eval, games, epochs: int, args, log=print) -> dict:
    import torch
    from az.model import build_model

    ids = train_ids if games is None else train_ids[:games]
    if games is not None and games > len(train_ids):
        raise SystemExit(f"学習に使える局は {len(train_ids)} 局しかありません (指定 {games} 局)")
    train = subset(data, ids)
    torch.manual_seed(args.seed)
    model = build_model({"arch": "token", "d_model": args.d_model, "layers": args.layers, "heads": args.heads})
    rows = []

    def on_epoch(model, info):
        test = test_eval(model)
        val = info["val"] or {}
        row = {"epoch": info["epoch"], "lr": info["lr"], "seconds": round(info["seconds"], 1),
               "train_loss": info["train_loss"],
               "val_loss": val.get("loss"), "val_value": val.get("value"), "val_policy": val.get("policy"),
               "test_loss": test["loss"], "test_value": test["value"], "test_policy": test["policy"],
               "test_acc": test["acc"] * 100}
        rows.append(row)
        log(f"    評価用: 損失 {test['loss']:.4f} (価値 {test['value']:.4f}, 方策 {test['policy']:.4f}), "
            f"一致率 {test['acc'] * 100:.1f}%, 学習率 {info['lr']:.2e}")

    start = time.perf_counter()
    fit(model, train, out=None, epochs=epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, value_weight=args.value_weight, augment=True, val_ratio=args.val_ratio,
        device=args.device, seed=args.seed, keep_best=False, amp=args.amp, log=log, on_epoch=on_epoch)
    return {"games": len(ids), "positions": len(train["values"]), "epochs": epochs,
            "seconds": round(time.perf_counter() - start, 1), "best_epoch": best_epochs(rows), "rows": rows}


def format_summary(results: list) -> str:
    lines = ["局数     局面数  エポック | 最小の損失 (エポック)  方策 (エポック)  価値 (エポック)  一致率 (エポック) | 最後の損失"]
    for r in results:
        rows = {row["epoch"]: row for row in r["rows"]}
        b = r["best_epoch"]
        last = r["rows"][-1]
        lines.append(f"{r['games']:6d} {r['positions']:9d} {r['epochs']:5d}    | "
                     f"{rows[b['loss']]['test_loss']:.4f} ({b['loss']:3d})        "
                     f"{rows[b['policy']]['test_policy']:.4f} ({b['policy']:3d})  "
                     f"{rows[b['value']]['test_value']:.4f} ({b['value']:3d})  "
                     f"{rows[b['acc']]['test_acc']:5.1f}% ({b['acc']:3d})     | {last['test_loss']:.4f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="事前学習のエポック数の検証")
    parser.add_argument("--data", nargs="+", required=True, help="棋譜のディレクトリ (複数可)")
    parser.add_argument("--runs", nargs="+", required=True, help="実験の一覧. 各要素は「局数x エポック数」(例: 30000x30, allx10)")
    parser.add_argument("--test-games", type=int, default=1000, help="評価用に取り分ける局数 (どの学習にも使わない)")
    parser.add_argument("--out", required=True, help="結果を保存するディレクトリ")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3, help="学習率の最大値 (OneCycle)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.05, help="学習用の局から fit が分ける検証用の割合")
    parser.add_argument("--amp", action="store_true", help="混合精度 (bf16) で学習する")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (省略時は GPU があれば cuda)")
    parser.add_argument("--seed", type=int, default=0, help="ネットワークの初期値と評価用の局の選び方のシード")
    args = parser.parse_args()

    import torch
    from az.model import default_device
    args.device = str(torch.device(args.device) if args.device else default_device())
    runs = [parse_run(r) for r in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data)
    test, train_ids = split_games(data, args.test_games, args.seed)
    print(f"棋譜: {len(np.unique(data['games']))} 局 ({len(data['values'])} 局面). "
          f"評価用 {args.test_games} 局 ({len(test['values'])} 局面), 学習に使える局 {len(train_ids)} 局", flush=True)
    test_eval = make_evaluator(test, torch.device(args.device), args.value_weight)

    results = []
    for games, epochs in runs:
        label = f"{'all' if games is None else games}x{epochs}"
        print(f"===== 実験 {label} =====", flush=True)
        result = run_one(data, train_ids, test_eval, games, epochs, args, log=lambda t: print(t, flush=True))
        result["label"] = label
        results.append(result)
        with open(out / f"run_{label}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
        summary = {"args": vars(args), "test_positions": len(test["values"]), "results": results}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"実験 {label}: {result['seconds']:.0f} 秒. 最も良かったエポック {result['best_epoch']}", flush=True)

    print("\n" + format_summary(results))
    print(f"\n結果: {out}")


if __name__ == "__main__":
    main()
