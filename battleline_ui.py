"""バトルラインの対戦プログラム (play_vs_random.py, play_vs_player.py) で共有する表示と入力の処理"""
import unicodedata

import pybl

FORMATION_NAMES = {
    pybl.WEDGE: "ウェッジ",
    pybl.PHALANX: "ファランクス",
    pybl.BATTALION: "バタリオン",
    pybl.SKIRMISHER: "スカーミッシャー",
    pybl.HOST: "ホスト",
}

ANSI_COLORS = {
    pybl.RED: "\033[91m",
    pybl.YELLOW: "\033[93m",
    pybl.BLUE: "\033[94m",
    pybl.GREEN: "\033[92m",
    pybl.PURPLE: "\033[95m",
    pybl.ORANGE: "\033[38;5;208m",
}
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"
ANSI_CLEAR = "\033[2J\033[H"

HELP = """\
着手の入力 (フラッグ番号は 1 - 9):
  R5 3    カード R5 をフラッグ 3 に置く ("R5@3" でも可)
  2 3     手札の 2 番目のカードをフラッグ 3 に置く
  pass    パス (置ける場所が無い場合のみ)
  b       盤面を再表示
  h       このヘルプ
  q       終了
フラッグ番号の横の * は、あなたがカードを置けるフラッグです。
カードは 色 + 数字 で表します: R=赤, Y=黄, B=青, G=緑, P=紫, O=オレンジ.
"""


class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def card(self, card) -> str:
        text = pybl.card_to_str(card)
        if not self.enabled:
            return text
        return f"{ANSI_COLORS[pybl.card_color(card)]}{text}{ANSI_RESET}"

    def bold(self, text: str) -> str:
        return f"{ANSI_BOLD}{text}{ANSI_RESET}" if self.enabled else text

    def dim(self, text: str) -> str:
        return f"{ANSI_DIM}{text}{ANSI_RESET}" if self.enabled else text


def display_width(text: str) -> int:
    """全角文字を2文字分として数えた表示幅 (ANSI エスケープは除く)"""
    width, in_escape = 0, False
    for ch in text:
        if ch == "\033":
            in_escape = True
        elif in_escape:
            in_escape = ch != "m"
        else:
            width += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
    return width


def pad(text: str, width: int, align: str = "left") -> str:
    space = " " * max(0, width - display_width(text))
    return space + text if align == "right" else text + space


def sorted_cards(bits) -> list:
    return sorted(pybl.CardIterator(bits), key=lambda c: (pybl.card_color(c), pybl.card_rank(c)))


def cards_str(bits, style: Style) -> str:
    return "  ".join(style.card(c) for c in sorted_cards(bits))


def formation_cell(state, flag: int, player, style: Style) -> str:
    cards = state.get_flag_cards(flag, player)
    text = " ".join(style.card(c) for c in cards)
    if len(cards) == 3:
        strength = state.get_flag_strength(flag, player)
        name = FORMATION_NAMES[pybl.strength_to_formation_type(strength)]
        text += style.dim(f" ({name} {pybl.strength_to_sum(strength)})")
    return text


def render(obs: pybl.Observation, turn: int, style: Style, names: dict) -> str:
    """obs.player の視点で盤面を表示する. 左が相手, 右が obs.player. names はプレイヤー -> 表示名"""
    me = obs.player
    opp = pybl.to_opponent(me)
    placeable = obs.get_placeable_flags(me) if obs.is_my_turn else 0
    claimed = {me: obs.get_claimed_flags(me), opp: obs.get_claimed_flags(opp)}

    width = 30
    lines = [
        style.bold(f"=== {turn} 手目 ===") +
        f"  山札 {obs.deck_count} 枚 / {names[opp]}の手札 {obs.opponent_hand_count} 枚 / "
        f"確保数 {names[me]} {bin(claimed[me]).count('1')} - {bin(claimed[opp]).count('1')} {names[opp]}",
        f"{pad(names[opp], width, 'right')}  | No |  {names[me]}",
    ]
    for i in range(pybl.NUM_FLAGS):
        mark = "*" if placeable >> i & 1 else " "
        if claimed[opp] >> i & 1:
            status = f"◀ {names[opp]}が確保"
        elif claimed[me] >> i & 1:
            status = f"▶ {names[me]}が確保"
        else:
            status = ""
        left = formation_cell(obs, i, opp, style)
        right = formation_cell(obs, i, me, style)
        lines.append(f"{pad(left, width, 'right')}  | {i + 1}{mark} |  {pad(right, width)} "
                     f"{style.dim(status) if status else ''}".rstrip())

    hand = sorted_cards(obs.hand)
    lines.append("")
    lines.append(f"{names[me]}の手札: " +
                 "  ".join(f"{style.dim(f'[{k + 1}]')}{style.card(c)}" for k, c in enumerate(hand)))
    return "\n".join(lines)


def parse_input(text: str, obs: pybl.Observation):
    """入力を着手に変換し (着手, エラーメッセージ) を返す"""
    if text in ("pass", "p"):
        if obs.is_legal(pybl.PASS_MOVE):
            return pybl.PASS_MOVE, None
        return None, "置ける場所があるためパスはできません。"

    tokens = text.replace("@", " ").split()
    if len(tokens) != 2:
        return None, "「カード フラッグ番号」の形で入力してください (例: R5 3)。h でヘルプを表示します。"

    card_text, flag_text = tokens
    hand = sorted_cards(obs.hand)
    if card_text.isdigit():
        index = int(card_text) - 1
        if not 0 <= index < len(hand):
            return None, f"手札の番号は 1 - {len(hand)} で指定してください。"
        card = hand[index]
    else:
        card = pybl.parse_card_str(card_text)
        if card == pybl.NULL_CARD:
            return None, f"「{card_text}」はカードとして読み取れません (例: R5, O10)。"
        if card not in hand:
            return None, f"{pybl.card_to_str(card)} は手札にありません。"

    if not flag_text.isdigit() or not 1 <= int(flag_text) <= pybl.NUM_FLAGS:
        return None, "フラッグ番号は 1 - 9 で指定してください。"
    flag = int(flag_text) - 1

    if obs.get_flag_owner(flag) != pybl.NULL_PLAYER:
        return None, f"フラッグ {flag + 1} は既に確保されています。"
    if len(obs.get_flag_cards(flag, obs.player)) == 3:
        return None, f"フラッグ {flag + 1} にはもう置けません (3枚配置済み)。"

    move = pybl.make_move(card, flag)
    if not obs.is_legal(move):
        return None, "その手は指せません。"
    return move, None


def ask_move(obs: pybl.Observation, turn: int, style: Style, names: dict):
    """obs.player に着手を入力させる. 中断された場合は None を返す"""
    moves = obs.get_legal_moves()
    if len(moves) == 1 and moves[0] == pybl.PASS_MOVE:
        print("置ける場所が無いため、パスします。")
        return pybl.PASS_MOVE

    while True:
        try:
            text = input(style.bold(f"{names[obs.player]}の手 > ")).strip().lower()
        except EOFError:
            print()
            return None
        if text in ("q", "quit", "exit"):
            return None
        if text in ("h", "help", "?"):
            print(HELP)
            continue
        if text in ("b", "board"):
            print(render(obs, turn, style, names))
            continue
        move, error = parse_input(text, obs)
        if error:
            print(error)
            continue
        return move


def describe_claims(claimed: int, state: pybl.GameState, names: dict) -> list:
    return [f"  → フラッグ {i + 1} を{names[state.get_flag_owner(i)]}が確保しました。"
            for i in range(pybl.NUM_FLAGS) if claimed >> i & 1]


def result_message(state: pybl.GameState, names: dict) -> str:
    if state.winner == pybl.NULL_PLAYER:
        message = "引き分けです。"
    else:
        message = f"{names[state.winner]}の勝ちです!"
    if state.is_forced_termination:
        message += "\n(両者が続けてパスしたため、確保したフラッグの数で決着しました)"
    return message
