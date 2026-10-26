"""AlphaZero 型の強化学習のループ (自己対戦 → 学習 → 評価) を繰り返す.

1世代の流れ:
    1. 自己対戦: 最も強いネットワーク (best) と ISMCTS で games 局を行い, data/gen_XXXX に保存する
    2. 学習: 最新のネットワークから, 直近 window 世代分の自己対戦のデータで学習し, checkpoints/gen_XXXX.pt に保存する
    3. ゲーティング: 新しいネットワークと best を gate_games 試合対戦させ, 勝率が gate_threshold 以上なら best を更新する
       (gate_threshold が 0 なら対戦せずに常に更新する)
    4. 既存のエージェントとの比較: eval_every 世代ごとに, best を eval_agents と対戦させる
    ゲーティングと比較の対戦の az は, 自己対戦と同じ探索の設定 (c_puct, batch_size, fpu, fpu_reduction) を使う
    eval_agents には match.py と同じ指定を書ける (例: abdeck az@checkpoint=runs/az4/checkpoints/gen_0005.pt,c_puct=1.0)

    python -m az.loop --run runs/az1 --init checkpoints/pretrain.pt --iterations 20 --games 500

途中で止めても, 同じ --run を指定すれば続きから再開する (状態は runs/<名前>/state.json, 結果は log.csv).
CPU アフィニティは match.py と同じく, 環境変数 BL_AFFINITY (または --affinity) で指定できる.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # bl_ex のディレクトリ

from az import selfplay  # noqa: E402
from az.evaluate import evaluate, format_result  # noqa: E402


def _log(run: Path, text: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}"
    print(line, flush=True)
    with open(run / "loop.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _az_spec(checkpoint: Path, config: dict, simulations: int, device: str = "cpu") -> str:
    """対戦に使う az の指定. 探索の設定 (c_puct, batch_size, fpu, fpu_reduction) は自己対戦と同じにする
    (ノイズと温度は使わない)"""
    spec = (f"az@checkpoint={checkpoint},num_simulations={simulations},c_puct={config['c_puct']},"
            f"batch_size={config['batch_size']},fpu={config['fpu']},device={device}")
    if config.get("fpu_reduction") is not None:
        spec += f",fpu_reduction={config['fpu_reduction']}"
    return spec


def _append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="AlphaZero 型の強化学習のループ")
    parser.add_argument("--run", required=True, help="実行の保存先のディレクトリ (例: runs/az1). 既にあれば続きから再開する")
    parser.add_argument("--init", default=None, help="最初のネットワーク (教師あり学習の結果など). 省略時はランダムな初期値")
    parser.add_argument("--iterations", type=int, default=10, help="この実行で進める世代数")
    parser.add_argument("--games", type=int, default=500, help="1世代あたりの自己対戦の局数")
    parser.add_argument("--window", type=int, default=4, help="学習に使う直近の世代数 (リプレイバッファ)")
    parser.add_argument("--epochs", type=int, default=2, help="1世代あたりの学習のエポック数")
    parser.add_argument("--train-batch-size", type=int, default=1024, help="学習のバッチの大きさ")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=0.0, help="価値の正解に混ぜる探索の価値 q の割合")
    parser.add_argument("--gate-games", type=int, default=200, help="ゲーティングの対戦数")
    parser.add_argument("--gate-threshold", type=float, default=0.55, help="best を更新する勝率 (0 なら常に更新)")
    parser.add_argument("--eval-agents", nargs="*", default=["abdeck"], help="定期的に対戦させる既存のエージェント")
    parser.add_argument("--eval-every", type=int, default=5, help="既存のエージェントと対戦させる間隔 (世代数)")
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--eval-simulations", type=int, default=None, help="評価の対戦での az のシミュレーション回数")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="自己対戦と対戦の並列化数")
    parser.add_argument("--affinity", default=os.environ.get("BL_AFFINITY"))
    parser.add_argument("--device", default=None, help="学習に使う装置 (cuda / cpu)")
    parser.add_argument("--selfplay-device", default=None,
                        help="自己対戦の推論に使う装置 (cuda / cpu, 省略時は GPU があれば cuda)")
    parser.add_argument("--eval-device", default="cpu",
                        help="ゲーティングと評価の対戦で az の推論に使う装置 (cpu / cuda / auto)")
    parser.add_argument("--concurrency", type=int, default=0,
                        help="自己対戦の1プロセスで同時に進める局の数 (0 なら cuda で 32, cpu で 1)")
    selfplay.add_config_arguments(parser)
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from match import apply_cpu_affinity
    from az.model import build_model, load_checkpoint, save_checkpoint
    from az.train import fit, load_dataset

    run = Path(args.run).resolve()
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "data").mkdir(parents=True, exist_ok=True)
    state_path = run / "state.json"
    config = {key: getattr(args, key) for key in selfplay.DEFAULT_CONFIG}
    eval_sims = args.eval_simulations or config["num_simulations"]

    # アフィニティは最初に1回だけ設定する (以降に作る自己対戦・対戦のプロセスに引き継がれる)
    cpus = apply_cpu_affinity(args.affinity)
    jobs = args.jobs or (len(cpus) if cpus else os.cpu_count()) or 1

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _log(run, f"再開: 世代 {state['generation']} (best は世代 {state['best']})")
    else:
        first = run / "checkpoints" / "gen_0000.pt"
        if args.init:
            shutil.copy(args.init, first)
            _log(run, f"新規: 最初のネットワークとして {args.init} を使う")
        else:
            save_checkpoint(first, build_model({"arch": "token", "d_model": 128, "layers": 4, "heads": 4}))
            _log(run, "新規: ランダムな初期値のネットワークから始める")
        state = {"generation": 0, "best": 0}
        (run / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
        state_path.write_text(json.dumps(state), encoding="utf-8")

    checkpoint = lambda g: run / "checkpoints" / f"gen_{g:04d}.pt"
    data_dir = lambda g: run / "data" / f"gen_{g:04d}"

    for _ in range(args.iterations):
        gen = state["generation"] + 1
        best = state["best"]
        row = {"generation": gen, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        _log(run, f"===== 世代 {gen} (自己対戦は世代 {best} のネットワーク) =====")

        # 1. 自己対戦 (既にデータがあれば, 途中で止めた世代なので作り直す)
        if data_dir(gen).exists():
            shutil.rmtree(data_dir(gen))
        stats = selfplay.generate(checkpoint(best), args.games, data_dir(gen), config, jobs=jobs, affinity=None,
                                  device=args.selfplay_device, concurrency=args.concurrency,
                                  log=lambda text: _log(run, text))
        row.update(selfplay_games=stats["games"], positions=stats["positions"], selfplay_seconds=round(stats["seconds"]))

        # 2. 学習 (最新のネットワークから, 直近 window 世代分のデータで)
        window = [data_dir(g) for g in range(max(1, gen - args.window + 1), gen + 1) if data_dir(g).exists()]
        model, _ = load_checkpoint(checkpoint(gen - 1), device="cpu")
        start = time.perf_counter()
        fit_result = fit(model, load_dataset(window), out=checkpoint(gen), epochs=args.epochs,
                         batch_size=args.train_batch_size, lr=args.lr, weight_decay=args.weight_decay,
                         value_weight=args.value_weight, q_weight=args.q_weight, device=args.device, seed=gen,
                         keep_best=False, log=lambda text: _log(run, text), kind="selfplay", generation=gen,
                         data=[str(d) for d in window])
        val = fit_result["val"] or {}
        row.update(train_seconds=round(time.perf_counter() - start), train_loss=round(fit_result["train_loss"], 4),
                   val_value_loss=round(val.get("value", float("nan")), 4),
                   val_policy_loss=round(val.get("policy", float("nan")), 4))

        # 3. ゲーティング
        if args.gate_threshold > 0:
            result = evaluate(_az_spec(checkpoint(gen), config, eval_sims, args.eval_device),
                              _az_spec(checkpoint(best), config, eval_sims, args.eval_device),
                              args.gate_games, jobs=jobs, seed=gen)
            accepted = result["rate"] >= args.gate_threshold
            _log(run, f"ゲーティング 世代 {gen} 対 世代 {best}: {format_result(result)} → "
                      f"{'best を更新' if accepted else 'best はそのまま'}")
            row.update(gate_rate=round(result["rate"], 4))
        else:
            accepted = True
            row.update(gate_rate="")
        if accepted:
            state["best"] = gen
        row.update(best=state["best"])

        # 4. 既存のエージェントとの比較
        for agent in args.eval_agents:
            key = f"vs_{agent.split('@')[0]}"
            if args.eval_every > 0 and gen % args.eval_every == 0:
                result = evaluate(_az_spec(checkpoint(state["best"]), config, eval_sims, args.eval_device), agent,
                                  args.eval_games, jobs=jobs, seed=10_000 + gen)
                _log(run, f"評価 世代 {state['best']} 対 {agent}: {format_result(result)}")
                row[key] = round(result["rate"], 4)
            else:
                row[key] = ""

        state["generation"] = gen
        state_path.write_text(json.dumps(state), encoding="utf-8")
        _append_csv(run / "log.csv", row)

    _log(run, f"終了: 世代 {state['generation']} まで進めた (best は世代 {state['best']})")


if __name__ == "__main__":
    main()
