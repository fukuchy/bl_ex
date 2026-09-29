#!/usr/bin/env python3
"""あなたとランダムエージェントでバトルライン (戦術カード無し) を対戦する.

使い方:
    python3 play_vs_random.py                 # 先手はランダム
    python3 play_vs_random.py --first you     # あなたが先手
    python3 play_vs_random.py --first agent   # エージェントが先手
    python3 play_vs_random.py --seed 42       # 配札を固定する
    python3 play_vs_random.py --no-color      # 色を付けない

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


def main():
    parser = argparse.ArgumentParser(description="あなたとランダムエージェントでバトルラインを対戦する")
    parser.add_argument("--seed", type=int, default=None, help="配札の乱数シード")
    parser.add_argument("--first", choices=["you", "agent", "random"], default="random", help="先手")
    parser.add_argument("--no-color", action="store_true", help="色を付けない")
    args = parser.parse_args()

    style = Style(enabled=not args.no_color and sys.stdout.isatty())
    rng = random.Random(args.seed)
    state = pybl.GameState(seed=args.seed)

    if args.first == "you":
        me = state.first_player
    elif args.first == "agent":
        me = pybl.to_opponent(state.first_player)
    else:
        me = rng.choice([pybl.FIRST, pybl.SECOND])
    agent = pybl.to_opponent(me)
    names = {me: "あなた", agent: "エージェント"}

    print(style.bold("バトルライン (戦術カード無し) — あなた vs ランダムエージェント"))
    print(f"あなたは{'先手' if me == state.first_player else '後手'}です。h でヘルプを表示します。\n")

    turn = 1
    while not state.is_terminal:
        if state.side_to_move == me:
            obs = state.observe(me)
            print(render(obs, turn, style, names))
            move = ask_move(obs, turn, style, names)
            if move is None:
                print("対局を中断しました。")
                return
        else:
            # エージェントも自分から見える情報のみを使って着手を選ぶ
            moves = state.observe(agent).get_legal_moves()
            move = moves[rng.randrange(len(moves))]

        player = state.side_to_move
        claimed = state.play(move)
        print(f"{names[player]}: {pybl.move_to_str(move)}")
        for message in describe_claims(claimed, state, names):
            print(style.bold(message))
        print()
        turn += 1

    print(render(state.observe(me), turn, style, names))
    print(f"エージェントの手札: {cards_str(state.get_hand(agent), style) or '(なし)'}\n")
    print(style.bold(result_message(state, names)))


if __name__ == "__main__":
    main()
