#!/usr/bin/env python3
"""1台のターミナルでプレイヤー同士がバトルライン (戦術カード無し) を対戦する.

手札は相手に見せてはいけないため, 手番の交代ごとに画面を消して交代画面を表示します.
交代画面が出たら, 手番のプレイヤーだけが画面を見て Enter を押してください.

使い方:
    python3 play_vs_player.py                          # 先手はランダム
    python3 play_vs_player.py --names たろう はなこ     # プレイヤー名を指定する
    python3 play_vs_player.py --first 1                # プレイヤー1 が先手 (2 ならプレイヤー2)
    python3 play_vs_player.py --seed 42                # 配札を固定する
    python3 play_vs_player.py --open                   # 手札を隠さない (交代画面なし, 両者の手札を表示)
    python3 play_vs_player.py --no-color               # 色を付けない

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

from battleline_ui import ANSI_CLEAR, Style, ask_move, cards_str, describe_claims, render, result_message


class Screen:
    def __init__(self, clear_enabled: bool):
        self.clear_enabled = clear_enabled

    def clear(self):
        if self.clear_enabled:
            print(ANSI_CLEAR, end="", flush=True)

    @staticmethod
    def wait(prompt: str) -> bool:
        """Enter を待つ. 入力が終了した場合や q が入力された場合は False を返す"""
        try:
            return input(prompt).strip().lower() not in ("q", "quit", "exit")
        except EOFError:
            print()
            return False


def main():
    parser = argparse.ArgumentParser(description="1台のターミナルでプレイヤー同士がバトルラインを対戦する")
    parser.add_argument("--names", nargs=2, metavar=("NAME1", "NAME2"), default=["プレイヤー1", "プレイヤー2"],
                        help="プレイヤー1 とプレイヤー2 の名前")
    parser.add_argument("--first", choices=["1", "2", "random"], default="random", help="先手のプレイヤー")
    parser.add_argument("--seed", type=int, default=None, help="配札の乱数シード")
    parser.add_argument("--open", action="store_true", help="手札を隠さない (交代画面なし)")
    parser.add_argument("--no-color", action="store_true", help="色を付けない")
    args = parser.parse_args()

    is_tty = sys.stdout.isatty()
    style = Style(enabled=not args.no_color and is_tty)
    screen = Screen(clear_enabled=is_tty and not args.open)
    rng = random.Random(args.seed)
    state = pybl.GameState(seed=args.seed)

    # プレイヤー1 が先手 (state.first_player) か後手かを決める
    player1_first = {"1": True, "2": False}.get(args.first, rng.random() < 0.5)
    player1 = state.first_player if player1_first else pybl.to_opponent(state.first_player)
    names = {player1: args.names[0], pybl.to_opponent(player1): args.names[1]}

    screen.clear()
    print(style.bold("バトルライン (戦術カード無し) — プレイヤー対戦"))
    print(f"先手: {names[state.first_player]} / 後手: {names[pybl.to_opponent(state.first_player)]}")
    if not args.open:
        print("手番の交代ごとに画面が消えます。交代画面では、手番のプレイヤーだけが画面を見てください。")
    print("h でヘルプを表示します。\n")

    turn = 1
    last_report = []
    while not state.is_terminal:
        player = state.side_to_move
        opponent = pybl.to_opponent(player)

        if not args.open:
            print(style.bold(f"{names[player]}の番です。") + f"{names[opponent]}は画面を見ないでください。")
            if not screen.wait(f"{names[player]}の準備ができたら Enter を押してください > "):
                print("対局を中断しました。")
                return
            screen.clear()

        if last_report:
            print("\n".join(last_report) + "\n")

        obs = state.observe(player)
        print(render(obs, turn, style, names))
        if args.open:
            print(f"{names[opponent]}の手札: {cards_str(state.get_hand(opponent), style) or '(なし)'}")

        move = ask_move(obs, turn, style, names)
        if move is None:
            print("対局を中断しました。")
            return

        claimed = state.play(move)
        last_report = [f"{names[player]}: {pybl.move_to_str(move)}"]
        last_report += [style.bold(m) for m in describe_claims(claimed, state, names)]
        print("\n".join(last_report) + "\n")
        turn += 1

        if not args.open and not state.is_terminal:
            if not screen.wait(f"Enter を押して画面を隠し、{names[opponent]}に交代してください > "):
                print("対局を中断しました。")
                return
            screen.clear()

    # 終局後は両者の手札を公開する
    first = state.first_player
    print(render(state.observe(first), turn, style, names))
    second = pybl.to_opponent(first)
    print(f"{names[second]}の手札: {cards_str(state.get_hand(second), style) or '(なし)'}")
    print()
    print(style.bold(result_message(state, names)))


if __name__ == "__main__":
    main()
