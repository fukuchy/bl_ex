#!/usr/bin/env python3
"""エージェント同士を指定回数対戦させ, 成績を表示する.

使い方:
    python3 match.py 試合数 エージェント1 エージェント2 [--first {1,2,alt}] [--seed N] [--quiet]

    python3 match.py 100 ab random                # 先手は交互 (既定)
    python3 match.py 100 ab random --first 1      # 常にエージェント1が先手
    python3 match.py 100 ab random --first 2      # 常にエージェント2が先手
    python3 match.py 100 ab ab --seed 0           # 同じ対局を再現できるようにする

エージェントの指定:
    random                    agent_random.random_move
    ab                        agent_ab.alphabeta_move
    モジュール名:関数名        例: agent_ab:alphabeta_move

エージェントは「局面を受け取って着手を返す関数」です. 公平のため, 手番のプレイヤーから見える情報
(pybl.Observation) のみを渡します. 関数が rng 引数を受け取る場合は, 試合ごとに random.Random を渡します.
"""
import argparse
import importlib
import inspect
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pybl

# 短い名前で指定できるエージェント
AGENTS = {
    "random": ("agent_random", "random_move"),
    "ab": ("agent_ab", "alphabeta_move"),
}


@dataclass
class Agent:
    name: str
    func: object
    accepts_rng: bool
    wins: int = 0
    losses: int = 0
    draws: int = 0
    first_record: list = field(default_factory=lambda: [0, 0, 0])   # 先手番での [勝ち, 負け, 引き分け]
    second_record: list = field(default_factory=lambda: [0, 0, 0])  # 後手番での [勝ち, 負け, 引き分け]
    move_count: int = 0
    think_time: float = 0.0

    def choose(self, obs: pybl.Observation, rng: random.Random):
        start = time.perf_counter()
        move = self.func(obs, rng=rng) if self.accepts_rng else self.func(obs)
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


def load_agent(spec: str) -> Agent:
    module_name, sep, func_name = spec.partition(":")
    if not sep:
        if spec not in AGENTS:
            raise SystemExit(f"不明なエージェントです: {spec} (指定できる名前: {', '.join(AGENTS)}, "
                             f"または モジュール名:関数名)")
        module_name, func_name = AGENTS[spec]

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise SystemExit(f"エージェントのモジュールを読み込めません: {module_name} ({e})")
    func = getattr(module, func_name, None)
    if not callable(func):
        raise SystemExit(f"{module_name} に関数 {func_name} がありません")

    accepts_rng = "rng" in inspect.signature(func).parameters
    return Agent(name=spec, func=func, accepts_rng=accepts_rng)


def play_game(agents: dict, state: pybl.GameState, rngs: dict) -> None:
    """agents, rngs はプレイヤー (pybl.FIRST / pybl.SECOND) -> Agent / random.Random"""
    while not state.is_terminal:
        player = state.side_to_move
        obs = state.observe(player)
        move = agents[player].choose(obs, rngs[player])
        if not state.is_legal(move):
            raise RuntimeError(f"{agents[player].name} が非合法手を返しました: {move!r} ({pybl.move_to_str(move)})\n{state}")
        state.play(move)


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
    parser.add_argument("agent1", help="エージェント1 (random, ab, または モジュール名:関数名)")
    parser.add_argument("agent2", help="エージェント2 (random, ab, または モジュール名:関数名)")
    parser.add_argument("--first", choices=["1", "2", "alt"], default="alt",
                        help="先手: 1 = エージェント1, 2 = エージェント2, alt = 交互 (既定, 1試合目はエージェント1が先手)")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード (指定すると同じ対局を再現できる)")
    parser.add_argument("--quiet", action="store_true", help="試合ごとの結果を表示しない")
    args = parser.parse_args()

    if args.games <= 0:
        parser.error("試合数は1以上を指定してください")

    # エージェントのモジュールをこのファイルと同じディレクトリから読み込めるようにする
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    agent1, agent2 = load_agent(args.agent1), load_agent(args.agent2)
    if agent1.name == agent2.name:
        agent1.name, agent2.name = f"{agent1.name}(1)", f"{agent2.name}(2)"

    master = random.Random(args.seed)
    total_moves = forced = 0
    start = time.perf_counter()

    for game in range(args.games):
        state = pybl.GameState(seed=master.getrandbits(64))
        agent1_first = args.first == "1" or (args.first == "alt" and game % 2 == 0)
        first, second = state.first_player, pybl.to_opponent(state.first_player)
        agents = {first: agent1, second: agent2} if agent1_first else {first: agent2, second: agent1}
        rngs = {p: random.Random(master.getrandbits(64)) for p in (first, second)}

        play_game(agents, state, rngs)

        total_moves += state.move_count
        forced += state.is_forced_termination
        for player, agent in agents.items():
            outcome = 2 if state.winner == pybl.NULL_PLAYER else 0 if state.winner == player else 1
            agent.record(outcome, is_first=player == first)

        if not args.quiet:
            winner = "引き分け" if state.winner == pybl.NULL_PLAYER else f"{agents[state.winner].name} の勝ち"
            print(f"{game + 1:>{len(str(args.games))}}/{args.games}  先手: {agents[first].name:<12} "
                  f"{winner:<16} ({state.move_count} 手)", flush=True)

    print_summary(agent1, agent2, args.games, total_moves, forced, time.perf_counter() - start)


if __name__ == "__main__":
    main()
