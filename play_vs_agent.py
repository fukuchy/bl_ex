#!/usr/bin/env python3
"""あなたと, 指定したエージェントでバトルライン (戦術カード無し) を対戦する.

使い方:
    python3 play_vs_agent.py エージェント [--first {you,agent,random}] [--seed N] [--no-color]

    python3 play_vs_agent.py ab                 # アルファベータ法のエージェントと対戦 (先手はランダム)
    python3 play_vs_agent.py dp --first you     # あなたが先手
    python3 play_vs_agent.py ismcts --first agent --seed 42
    python3 play_vs_agent.py abdeck@depth=3     # エージェントの引数を指定する (@引数名=値, 複数はカンマ区切り)

エージェントの指定 (match.py と同じ):
    random, ab, mon, ismcts, root, dp, abdeck, beam, oracle, ismcts_eval, oracle_eval,
    または モジュール名:関数名 (例: agent_ab:alphabeta_move)
    後ろに @引数名=値 を付けると, エージェントの関数に引数を渡せる (例: abdeck@depth=3,num_samples=16)

エージェントには公平のため, エージェントから見える情報 (pybl.Observation) のみを渡します.

着手の入力 (フラッグ番号は 1 - 9):
    R5 3    カード R5 をフラッグ 3 に置く ("R5@3" でも可)
    2 3     手札の 2 番目のカードをフラッグ 3 に置く
    pass    パス (置ける場所が無い場合のみ)
    b       盤面を再表示
    h       ヘルプ
    q       終了

カードは 色 + 数字 で表します: R=赤, Y=黄, B=青, G=緑, P=紫, O=オレンジ.
"""
import argparse
import random
import sys

import pybl

from battleline_ui import Style, ask_move, cards_str, describe_claims, render, result_message
from match import AGENTS, load_agent


def main():
    parser = argparse.ArgumentParser(description="あなたと指定したエージェントでバトルラインを対戦する")
    parser.add_argument("agent", help=f"対戦するエージェント ({', '.join(AGENTS)}, または モジュール名:関数名. "
                                      f"@引数名=値 で引数を指定)")
    parser.add_argument("--first", choices=["you", "agent", "random"], default="random", help="先手 (既定はランダム)")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード (配札とエージェントの乱数を固定する)")
    parser.add_argument("--no-color", action="store_true", help="色を付けない")
    args = parser.parse_args()

    agent = load_agent(args.agent)
    style = Style(enabled=not args.no_color and sys.stdout.isatty())
    rng = random.Random(args.seed)
    agent_rng = random.Random(rng.getrandbits(64))
    state = pybl.GameState(seed=rng.getrandbits(64))

    if args.first == "you":
        me = state.first_player
    elif args.first == "agent":
        me = pybl.to_opponent(state.first_player)
    else:
        me = rng.choice([pybl.FIRST, pybl.SECOND])
    opponent = pybl.to_opponent(me)
    names = {me: "あなた", opponent: agent.name}

    print(style.bold(f"バトルライン (戦術カード無し) — あなた vs {agent.name}"))
    print(f"あなたは{'先手' if me == state.first_player else '後手'}です。h でヘルプを表示します。\n")

    turn = 1
    while not state.is_terminal:
        player = state.side_to_move
        obs = state.observe(player)
        if player == me:
            print(render(obs, turn, style, names))
            move = ask_move(obs, turn, style, names)
            if move is None:
                print("対局を中断しました。")
                return
            note = ""
        else:
            print(f"{agent.name} が考えています...", end="", flush=True)
            elapsed = agent.think_time
            move = agent.choose(obs, agent_rng, state)
            elapsed = agent.think_time - elapsed
            print("\r" + " " * 40 + "\r", end="")
            if not state.is_legal(move):
                raise RuntimeError(f"{agent.name} が非合法手を返しました: {move!r}")
            note = f"  ({elapsed:.2f} 秒)"

        claimed = state.play(move)
        print(f"{names[player]}: {pybl.move_to_str(move)}{note}")
        for message in describe_claims(claimed, state, names):
            print(style.bold(message))
        print()
        turn += 1

    print(render(state.observe(me), turn, style, names))
    print(f"{agent.name} の手札: {cards_str(state.get_hand(opponent), style) or '(なし)'}\n")
    print(style.bold(result_message(state, names)))


if __name__ == "__main__":
    main()
