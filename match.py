#!/usr/bin/env python3
"""エージェント同士を指定回数対戦させ, 成績を表示する.

使い方:
    python3 match.py 試合数 エージェント1 エージェント2 [--first {1,2,alt}] [--seed N] [--interval N] [--quiet] [-j N]

    python3 match.py 100 ab random                # 先手は交互 (既定)
    python3 match.py 100 ab random --first 1      # 常にエージェント1が先手
    python3 match.py 100 ab random --first 2      # 常にエージェント2が先手
    python3 match.py 100 ab ab --seed 0           # 同じ対局を再現できるようにする
    python3 match.py 100 ab dp -j 4               # 4試合ずつ並列に対戦させる (-j 0 で CPU のコア数)
    python3 match.py 100 ab dp --interval 10      # 10試合ごとに途中経過 (通算成績) を表示する
    python3 match.py 100 ab dp -j 8 --affinity 5555   # 論理プロセッサ 0,2,4,...,14 だけで実行する

CPU アフィニティ (使う論理プロセッサの指定):
    環境変数 BL_AFFINITY に16進数のマスクを設定しておくと, 毎回自動で適用される (--affinity で上書き,
    --affinity none で無効). 並列実行の子プロセスにも引き継がれる. Windows と Linux で有効 (macOS では無視).
    例: Core i9-12900 で各 P コアの1スレッド目 (論理プロセッサ 0, 2, ..., 14) だけを使う場合は 5555

エージェントの指定:
    random                    agent_random.random_move
    ab                        agent_ab.alphabeta_move
    mon                       agent_mon.montecarlo_move
    ismcts                    agent_ismcts.ismcts_move
    root                      agent_root.root_ismcts_move
    dp                        agent_dp.deep_barca_move
    abdeck                    agent_abDeck.abdeck_move
    beam                      agent_beam.beam_move
    oracle                    agent_oracle.oracle_move (相手の手札と山札を見る完全情報 MCTS. 比較用)
    ismcts_eval               agent_ismcts_eval.ismcts_eval_move (打ち切り＋静的評価の ISMCTS)
    oracle_eval               agent_ismcts_eval.oracle_eval_move (打ち切り＋静的評価の完全情報 MCTS. 比較用)
    azp                       agent_az.az_policy_move (学習したネットワークの方策だけで手を選ぶ)
    az                        agent_az.az_move (学習したネットワークを使う ISMCTS)
    モジュール名:関数名        例: agent_ab:alphabeta_move

エージェントの引数の指定 (エージェント名の後ろに @引数名=値 を付ける. 複数ある場合はカンマで区切る):
    abdeck@depth=3                    探索の深さを 3 にする
    abdeck@depth=3,num_samples=16     探索の深さを 3, 生成する局面の数を 16 にする
    mon@num_playouts=10000            プレイアウトの総数を 10000 にする
    beam@depth=4,num_samples=4,beam_width=5   ビームサーチの深さ・サンプル数・ビーム幅を指定する

エージェントは「局面を受け取って着手を返す関数」です. 公平のため, 手番のプレイヤーから見える情報
(pybl.Observation) のみを渡します. 関数が rng 引数を受け取る場合は, 試合ごとに random.Random を渡します.
ただし関数が state 引数を受け取る場合 (答えを見たエージェント oracle など) は, 比較実験のために
相手の手札と山札を含む本物の局面 (pybl.GameState) のコピーも渡します.
"""
import argparse
import ctypes
import importlib
import os
import inspect
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pybl

# 短い名前で指定できるエージェント
AGENTS = {
    "random": ("agent_random", "random_move"),
    "ab": ("agent_ab", "alphabeta_move"),
    "mon": ("agent_mon", "montecarlo_move"),
    "ismcts": ("agent_ismcts", "ismcts_move"),
    "root": ("agent_root", "root_ismcts_move"),
    "dp": ("agent_dp", "deep_barca_move"),
    "abdeck": ("agent_abDeck", "abdeck_move"),
    "beam": ("agent_beam", "beam_move"),
    "oracle": ("agent_oracle", "oracle_move"),
    "ismcts_eval": ("agent_ismcts_eval", "ismcts_eval_move"),
    "oracle_eval": ("agent_ismcts_eval", "oracle_eval_move"),
    "azp": ("agent_az", "az_policy_move"),
    "az": ("agent_az", "az_move"),
    "azo": ("agent_az", "az_oracle_move"),
}


@dataclass
class Agent:
    name: str
    func: object
    accepts_rng: bool
    accepts_state: bool = False                                     # 本物の局面を受け取るか (答えを見たエージェント用)
    kwargs: dict = field(default_factory=dict)                      # 着手のたびに関数へ渡す引数
    wins: int = 0
    losses: int = 0
    draws: int = 0
    first_record: list = field(default_factory=lambda: [0, 0, 0])   # 先手番での [勝ち, 負け, 引き分け]
    second_record: list = field(default_factory=lambda: [0, 0, 0])  # 後手番での [勝ち, 負け, 引き分け]
    move_count: int = 0
    think_time: float = 0.0

    def choose(self, obs: pybl.Observation, rng: random.Random, state: pybl.GameState | None = None):
        """着手を選ばせる. 関数が state 引数を受け取る場合に限り, 本物の局面のコピーを渡す"""
        extra = {}
        if self.accepts_rng:
            extra["rng"] = rng
        if self.accepts_state:
            extra["state"] = state.copy() if state is not None else None
        start = time.perf_counter()
        move = self.func(obs, **extra, **self.kwargs)
        self.think_time += time.perf_counter() - start
        self.move_count += 1
        return move

    def record(self, outcome: int, is_first: bool):
        """outcome: 0 = 勝ち, 1 = 負け, 2 = 引き分け"""
        (self.first_record if is_first else self.second_record)[outcome] += 1
        if outcome == 0:
            self.wins += 1
        elif outcome == 1:
            self.losses += 1
        else:
            self.draws += 1


def _parse_value(text: str):
    """引数の値を int, float, 文字列の順に解釈する"""
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            pass
    return text


def load_agent(spec: str) -> Agent:
    """エージェントの指定 (名前, モジュール名:関数名, 後ろに @引数名=値,... を付けてもよい) から Agent を作る"""
    target, _, params = spec.partition("@")
    kwargs = {}
    for item in filter(None, params.split(",")):
        key, eq, value = item.partition("=")
        if not eq or not key:
            raise SystemExit(f"エージェントの引数は 引数名=値 の形で指定してください: {item}")
        kwargs[key.strip()] = _parse_value(value.strip())

    module_name, sep, func_name = target.partition(":")
    if not sep:
        if target not in AGENTS:
            raise SystemExit(f"不明なエージェントです: {target} (指定できる名前: {', '.join(AGENTS)}, "
                             f"または モジュール名:関数名)")
        module_name, func_name = AGENTS[target]

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise SystemExit(f"エージェントのモジュールを読み込めません: {module_name} ({e})")
    func = getattr(module, func_name, None)
    if not callable(func):
        raise SystemExit(f"{module_name} に関数 {func_name} がありません")

    parameters = inspect.signature(func).parameters
    reserved = ("obs", "rng", "state")     # match.py が渡す引数 (@ では指定できない)
    fixed = [key for key in kwargs if key in reserved]
    if fixed:
        raise SystemExit(f"引数 {', '.join(fixed)} は match.py が渡すため, @ では指定できません")
    unknown = [key for key in kwargs if key not in parameters]
    if unknown:
        accepted = [name for name in parameters if name not in reserved]
        raise SystemExit(f"{func_name} は引数 {', '.join(unknown)} を受け取りません "
                         f"(指定できる引数: {', '.join(accepted) or 'なし'})")
    return Agent(name=spec, func=func, accepts_rng="rng" in parameters, accepts_state="state" in parameters,
                 kwargs=kwargs)


def play_game(agents: dict, state: pybl.GameState, rngs: dict) -> None:
    """agents, rngs はプレイヤー (pybl.FIRST / pybl.SECOND) -> Agent / random.Random"""
    while not state.is_terminal:
        player = state.side_to_move
        obs = state.observe(player)
        move = agents[player].choose(obs, rngs[player], state)
        if not state.is_legal(move):
            raise RuntimeError(f"{agents[player].name} が非合法手を返しました: {move!r} ({pybl.move_to_str(move)})\n{state}")
        state.play(move)


@dataclass
class GameTask:
    """1試合の条件. 並列実行のため, 別のプロセスに渡せる値だけを持つ"""
    index: int
    agent1_spec: str
    agent2_spec: str
    agent1_first: bool
    state_seed: int
    first_seed: int     # 先手のエージェントに渡す乱数のシード
    second_seed: int    # 後手のエージェントに渡す乱数のシード


@dataclass
class GameResult:
    index: int
    agent1_first: bool
    winner: int         # 1 = エージェント1の勝ち, 2 = エージェント2の勝ち, 0 = 引き分け
    move_count: int
    forced: bool
    think_time: tuple   # (エージェント1, エージェント2) の思考時間の合計 (秒)
    agent_moves: tuple  # (エージェント1, エージェント2) の着手数


# 各プロセスで読み込んだエージェント (エージェント1/2 の別ごとに1つ)
_loaded_agents = {}


def _agent_for(slot: int, spec: str) -> Agent:
    key = (slot, spec)
    if key not in _loaded_agents:
        _loaded_agents[key] = load_agent(spec)
    return _loaded_agents[key]


def run_game(task: GameTask) -> GameResult:
    """1試合を行い, 結果を返す (並列実行では別のプロセスで呼ばれる)"""
    agent1, agent2 = _agent_for(1, task.agent1_spec), _agent_for(2, task.agent2_spec)
    before = [(a.think_time, a.move_count) for a in (agent1, agent2)]

    state = pybl.GameState(seed=task.state_seed)
    first, second = state.first_player, pybl.to_opponent(state.first_player)
    agents = {first: agent1, second: agent2} if task.agent1_first else {first: agent2, second: agent1}
    rngs = {first: random.Random(task.first_seed), second: random.Random(task.second_seed)}
    play_game(agents, state, rngs)

    if state.winner == pybl.NULL_PLAYER:
        winner = 0
    else:
        winner = 1 if agents[state.winner] is agent1 else 2
    return GameResult(
        index=task.index, agent1_first=task.agent1_first, winner=winner, move_count=state.move_count,
        forced=bool(state.is_forced_termination),
        think_time=tuple(a.think_time - t for a, (t, _) in zip((agent1, agent2), before)),
        agent_moves=tuple(a.move_count - m for a, (_, m) in zip((agent1, agent2), before)),
    )


def apply_cpu_affinity(spec: str | None) -> list | None:
    """CPU アフィニティ (このプロセスと子プロセスが使う論理プロセッサ) を設定し, 使う論理プロセッサの番号を返す.

    spec は16進数のマスク (例: "5555", "0x5555"). None や "none" なら何もしない.
    設定できない OS (macOS など) では警告を表示して無視する.
    """
    if not spec or spec.strip().lower() == "none":
        return None
    try:
        mask = int(spec.strip(), 16)
    except ValueError:
        raise SystemExit(f"CPU アフィニティは16進数のマスクで指定してください: {spec}")
    if mask <= 0:
        raise SystemExit(f"CPU アフィニティのマスクは0より大きい値を指定してください: {spec}")
    cpus = [i for i in range(mask.bit_length()) if mask >> i & 1]

    if sys.platform == "win32":
        # Windows では子プロセスが親プロセスのアフィニティを引き継ぐため, 並列実行のプロセスにも適用される
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
            raise SystemExit(f"CPU アフィニティを設定できませんでした (マスク {spec}, エラー {ctypes.get_last_error()}). "
                             f"この PC に無い論理プロセッサを指定していないか確認してください")
    elif hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, cpus)
        except OSError as e:
            raise SystemExit(f"CPU アフィニティを設定できませんでした (マスク {spec}): {e}")
    else:
        print(f"この OS では CPU アフィニティを指定できないため, 指定 ({spec}) を無視します", file=sys.stderr)
        return None
    return cpus


def rate(wins: int, draws: int, games: int) -> str:
    return f"{(wins + 0.5 * draws) / games * 100:5.1f}%" if games else "   - "


def print_summary(agent1: Agent, agent2: Agent, games: int, total_moves: int, forced: int, elapsed: float):
    width = max(len(agent1.name), len(agent2.name), 8) + 2
    print(f"\n=== 結果 ({games} 試合, {elapsed:.1f} 秒) ===")
    print(f"{'':{width}} 勝ち  負け  引分  勝率 (引分は0.5勝)")
    for label, agent in (("1", agent1), ("2", agent2)):
        print(f"{label}: {agent.name:{width - 3}} {agent.wins:4d}  {agent.losses:4d}  {agent.draws:4d}  "
              f"{rate(agent.wins, agent.draws, games)}")

    print("\n先手番・後手番別の成績 (勝ち-負け-引分):")
    for label, agent in (("1", agent1), ("2", agent2)):
        f, s = agent.first_record, agent.second_record
        print(f"{label}: {agent.name:{width - 3}} 先手 {f[0]}-{f[1]}-{f[2]} ({rate(f[0], f[2], sum(f)).strip()})  "
              f"後手 {s[0]}-{s[1]}-{s[2]} ({rate(s[0], s[2], sum(s)).strip()})")

    print(f"\n平均手数: {total_moves / games:.1f}  /  両者連続パスによる強制終局: {forced} 試合")
    print("1手あたりの平均思考時間:")
    for label, agent in (("1", agent1), ("2", agent2)):
        average = agent.think_time / agent.move_count if agent.move_count else 0.0
        print(f"{label}: {agent.name:{width - 3}} {average * 1000:.2f} ms ({agent.move_count} 手)")


def main():
    parser = argparse.ArgumentParser(description="エージェント同士を指定回数対戦させる")
    parser.add_argument("games", type=int, help="試合数")
    parser.add_argument("agent1", help="エージェント1 (random, ab, mon, ismcts, root, dp, abdeck, beam, oracle, ismcts_eval, oracle_eval, azp, az, または モジュール名:関数名. @引数名=値 で引数を指定)")
    parser.add_argument("agent2", help="エージェント2 (random, ab, mon, ismcts, root, dp, abdeck, beam, oracle, ismcts_eval, oracle_eval, azp, az, または モジュール名:関数名. @引数名=値 で引数を指定)")
    parser.add_argument("--first", choices=["1", "2", "alt"], default="alt",
                        help="先手: 1 = エージェント1, 2 = エージェント2, alt = 交互 (既定, 1試合目はエージェント1が先手)")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード (指定すると同じ対局を再現できる)")
    parser.add_argument("--interval", type=int, default=1,
                        help="途中経過を表示する間隔 (試合数). 既定は1 (1試合ごとに結果を表示), "
                             "2以上なら指定した試合数ごとに通算成績を表示する")
    parser.add_argument("--quiet", action="store_true", help="途中経過を表示しない")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="同時に行う試合数 (並列化数). 既定は1 (並列化しない), 0 で使える論理プロセッサの数")
    parser.add_argument("--affinity", default=os.environ.get("BL_AFFINITY"),
                        help="使う論理プロセッサを16進数のマスクで指定する (例: 5555). "
                             "省略時は環境変数 BL_AFFINITY の値. none で指定しない")
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("試合数は1以上を指定してください")
    if args.jobs < 0:
        parser.error("並列化数は0以上を指定してください")
    if args.interval <= 0:
        parser.error("途中経過を表示する間隔は1以上を指定してください")
    # 並列実行のプロセスを作る前に設定する (子プロセスに引き継がれる)
    cpus = apply_cpu_affinity(args.affinity)
    if cpus is not None:
        print(f"CPU アフィニティ: 論理プロセッサ {', '.join(map(str, cpus))} を使用", flush=True)
    jobs = min(args.games, args.jobs or (len(cpus) if cpus else os.cpu_count()) or 1)

    # エージェントのモジュールをこのファイルと同じディレクトリから読み込めるようにする
    # (並列実行のプロセスにも sys.path が引き継がれる)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    agent1, agent2 = load_agent(args.agent1), load_agent(args.agent2)
    if agent1.name == agent2.name:
        agent1.name, agent2.name = f"{agent1.name}(1)", f"{agent2.name}(2)"

    # 各試合の乱数のシードを先に決めておくことで, 並列化数によらず同じシードなら同じ結果になる
    master = random.Random(args.seed)
    tasks = []
    for game in range(args.games):
        state_seed = master.getrandbits(64)
        first_seed, second_seed = master.getrandbits(64), master.getrandbits(64)
        agent1_first = args.first == "1" or (args.first == "alt" and game % 2 == 0)
        tasks.append(GameTask(game, args.agent1, args.agent2, agent1_first, state_seed, first_seed, second_seed))

    if jobs > 1:
        print(f"{jobs} 試合ずつ並列に対戦させます", flush=True)
    total_moves = forced = 0
    start = time.perf_counter()

    executor = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        # 試合の番号順に結果を受け取る (並列実行でも表示の順番は変わらない)
        results = executor.map(run_game, tasks) if executor else map(run_game, tasks)
        for result in results:
            total_moves += result.move_count
            forced += result.forced
            for slot, agent in ((1, agent1), (2, agent2)):
                outcome = 2 if result.winner == 0 else 0 if result.winner == slot else 1
                agent.record(outcome, is_first=result.agent1_first == (slot == 1))
                agent.think_time += result.think_time[slot - 1]
                agent.move_count += result.agent_moves[slot - 1]

            done = result.index + 1
            progress = f"{done:>{len(str(args.games))}}/{args.games}"
            if args.quiet:
                pass
            elif args.interval == 1:
                first_name = agent1.name if result.agent1_first else agent2.name
                winner = "引き分け" if result.winner == 0 else f"{(agent1, agent2)[result.winner - 1].name} の勝ち"
                print(f"{progress}  先手: {first_name:<12} {winner:<16} ({result.move_count} 手)", flush=True)
            elif done % args.interval == 0 or done == args.games:
                # 通算成績 (エージェント1から見た勝ち-負け-引分と勝率)
                print(f"{progress}  {agent1.name} {agent1.wins}勝 {agent1.losses}敗 {agent1.draws}分 "
                      f"({rate(agent1.wins, agent1.draws, done).strip()})  vs  {agent2.name}  "
                      f"[経過 {time.perf_counter() - start:.0f} 秒]", flush=True)
    finally:
        if executor:
            executor.shutdown(cancel_futures=True)

    print_summary(agent1, agent2, args.games, total_moves, forced, time.perf_counter() - start)


if __name__ == "__main__":
    main()
