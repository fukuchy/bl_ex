#!/usr/bin/env python3
"""あなたと, 指定したエージェントでバトルライン (戦術カード無し) を GUI (tkinter) で対戦する.

play_vs_agent.py の GUI 版. Python に付属の tkinter だけを使うので, 追加のインストールは不要.

使い方:
    python play_gui.py [エージェント] [--first {you,agent,random}] [--seed N]

    python play_gui.py abdeck
    python play_gui.py az@checkpoint=runs/az6/checkpoints/gen_0021.pt,c_puct=1.0,fpu_reduction=0.2,device=cuda

エージェントの指定は match.py と同じ (省略時は abdeck). ウィンドウの「新しい対局」で, 相手・先手・シードを
変えて対局し直せる.

操作:
    手札のカードをクリックして選び, 置けるフラッグの列 (青い枠が付いた列) をクリックする.
    カードを選んだあと数字キー 1 - 9 でも置ける. 同じカードをもう一度クリックするか Esc で選択を取り消す.
    置ける場所が無いときは自動でパスする. 置ける場所があってもパスしたいときは「パス」ボタン.

エージェントには公平のため, エージェントから見える情報 (pybl.Observation) のみを渡します
(state を受け取る答えを見たエージェントには, play_vs_agent.py と同じく本物の局面のコピーを渡す).
エージェントは別のスレッドで考えるので, 考えている間もウィンドウは固まらない.
"""
import argparse
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import pybl

from battleline_ui import FORMATION_NAMES, sorted_cards
from match import AGENTS, load_agent

DEFAULT_AGENT = "abdeck"
FIRST_CHOICES = ("you", "agent", "random")
FIRST_LABELS = {"random": "ランダム", "you": "あなた", "agent": "相手"}


def agent_label(spec: str) -> str:
    """画面に出すエージェントの短い名前 (az@checkpoint=.../gen_0021.pt,... → az (gen_0021))"""
    target, _, params = spec.partition("@")
    for item in params.split(","):
        key, _, value = item.partition("=")
        if key.strip() == "checkpoint" and value.strip():
            return f"{target} ({Path(value.strip()).stem})"
    return target


class Game:
    """1局分の状態と進行 (画面とは独立. エージェントを考えさせるのは呼び出し側)"""

    def __init__(self, agent_spec: str, first: str = "random", seed: int | None = None):
        if first not in FIRST_CHOICES:
            raise ValueError(f"先手の指定が不正です: {first}")
        try:
            self.agent = load_agent(agent_spec)
        except SystemExit as e:        # load_agent はエラーを SystemExit で知らせる
            raise ValueError(str(e)) from None
        self.spec = agent_spec
        self.label = agent_label(agent_spec)
        rng = random.Random(seed)
        self.agent_rng = random.Random(rng.getrandbits(64))
        self.state = pybl.GameState(seed=rng.getrandbits(64))
        if first == "you":
            self.me = self.state.first_player
        elif first == "agent":
            self.me = pybl.to_opponent(self.state.first_player)
        else:
            self.me = rng.choice([pybl.FIRST, pybl.SECOND])
        self.opp = pybl.to_opponent(self.me)
        self.log = []                  # (手数, "you" / "agent", 着手の文字列, 思考時間 or None, 注記, 確保したフラッグ)
        self.last_move = None          # 直前の着手 ("you" / "agent", 着手)
        self.error = None

    @property
    def turn(self) -> int:
        return len(self.log) + 1

    @property
    def my_turn(self) -> bool:
        return bool(not self.state.is_terminal and self.state.side_to_move == self.me and self.error is None)

    @property
    def agent_turn(self) -> bool:
        return bool(not self.state.is_terminal and self.state.side_to_move == self.opp and self.error is None)

    def legal_moves(self) -> list:
        """あなたの合法手 (あなたの手番でなければ空)"""
        return [int(m) for m in self.state.observe(self.me).get_legal_moves()] if self.my_turn else []

    def play(self, move, who: str, elapsed: float | None = None, note: str = "") -> list:
        """着手を進めて, 確保されたフラッグの [(番号 0-8, "you" / "agent")] を返す"""
        player = self.me if who == "you" else self.opp
        if self.state.is_terminal or self.state.side_to_move != player or not self.state.is_legal(move):
            raise ValueError(f"その手は指せません: {move!r}")
        claimed = self.state.play(move)
        claims = [(i, "you" if self.state.get_flag_owner(i) == self.me else "agent")
                  for i in range(pybl.NUM_FLAGS) if claimed >> i & 1]
        self.log.append((self.turn, who, pybl.move_to_str(move), elapsed, note, claims))
        self.last_move = (who, int(move))
        return claims

    def auto_pass(self) -> bool:
        """あなたの手番で, パスしか指せなければパスする (パスしたら True)"""
        if self.my_turn and self.legal_moves() == [int(pybl.PASS_MOVE)]:
            self.play(pybl.PASS_MOVE, "you", note="置ける場所が無いため自動でパス")
            return True
        return False

    def think(self) -> tuple:
        """エージェントに考えさせて (着手, 思考時間) を返す. 局面は変えないので, 別のスレッドから呼んでよい"""
        obs = self.state.observe(self.opp)
        before = self.agent.think_time
        move = self.agent.choose(obs, self.agent_rng, self.state.copy())
        return move, self.agent.think_time - before

    def formation(self, flag: int, who: str) -> str:
        """フラッグに 3 枚そろっていれば, 役の名前と合計 (例: ウェッジ 24)"""
        player = self.me if who == "you" else self.opp
        if len(self.state.get_flag_cards(flag, player)) < 3:
            return ""
        strength = self.state.get_flag_strength(flag, player)
        return f"{FORMATION_NAMES[pybl.strength_to_formation_type(strength)]} {pybl.strength_to_sum(strength)}"

    def result_text(self) -> str:
        winner = self.state.winner
        if winner == pybl.NULL_PLAYER:
            text = "引き分けです"
        elif winner == self.me:
            text = "あなたの勝ちです！"
        else:
            text = f"{self.label} の勝ちです"
        if self.state.is_forced_termination:
            text += "\n(両者が続けてパスしたため、確保したフラッグの数で決着)"
        return text


# ---------------------------------------------------------------------------
# 画面 (tkinter)
# ---------------------------------------------------------------------------

CARD_COLORS = {"R": "#d24a43", "Y": "#e3b51d", "B": "#3c6fd1", "G": "#2f9a5a", "P": "#8650c4", "O": "#e9782b"}
CARD_INK = {"Y": "#3b2f00"}
BG = "#f3f1ec"
FELT = "#e4dfd2"
FELT_TARGET = "#fbf7ec"
SLOT = "#d6d0c2"
INK = "#23211d"
MUTED = "#6f6a60"
ACCENT = "#2f6fb5"
YOU = "#2f6fb5"
AGENT = "#b5462f"
LAST = "#f2b705"

# 盤面の寸法 (ピクセル. 画面の拡大率が 100% のとき. それ以外では App.scale 倍に拡大する)
MARGIN = 16
COL_W = 84
COL_GAP = 8
CARD_H = 46
CARD_GAP = 5
TOP_H = 44                                      # 相手の手札の行
SIDE_H = 34 + 3 * (CARD_H + CARD_GAP)          # 役の名前と合計 (2 行) + カード 3 枚
FLAG_H = 56
HAND_W = 70
HAND_H = 100
BOARD_W = 2 * MARGIN + 9 * COL_W + 8 * COL_GAP
BOARD_TOP = MARGIN + TOP_H
FLAG_TOP = BOARD_TOP + SIDE_H
MY_TOP = FLAG_TOP + FLAG_H
BOARD_BOTTOM = MY_TOP + SIDE_H
HAND_TOP = BOARD_BOTTOM + 34
CANVAS_H = HAND_TOP + HAND_H + 16


def _pick_font(families) -> str:
    for name in ("Hiragino Sans", "Yu Gothic UI", "Meiryo UI", "Meiryo", "Noto Sans CJK JP"):
        if name in families:
            return name
    return "TkDefaultFont"


class App:
    """対局のウィンドウ"""

    def __init__(self, root, agent_spec: str, first: str, seed: int | None):
        import tkinter as tk
        import tkinter.messagebox
        from tkinter import font as tkfont, ttk

        self.tk, self.ttk, self.messagebox = tk, ttk, tkinter.messagebox
        self.root = root
        self.game = None
        self.game_id = 0
        self.selected = None          # 選んでいる手札のカード
        self.thinking_since = None
        self.results = queue.Queue()  # エージェントのスレッド → 画面: (game_id, 着手, 思考時間, エラー)

        # Windows の高 DPI の画面 (拡大率 125% など) では, 寸法と文字をその倍率で拡大する
        windows = root.tk.call("tk", "windowingsystem") == "win32"
        self.scale = max(1.0, root.winfo_fpixels("1i") / 96) if windows else 1.0
        self.family = _pick_font(set(tkfont.families(root)))
        self.font = self.px(14)
        self.font_bold = self.px(14, "bold")
        self.font_small = self.px(12)
        self.font_status = self.px(18, "bold")
        root.title("バトルライン")
        root.configure(bg=BG)

        self.canvas = tk.Canvas(root, width=round(BOARD_W * self.scale), height=round(CANVAS_H * self.scale), bg=BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="n", padx=(4, 0), pady=4)
        side = tk.Frame(root, bg=BG)
        side.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        root.grid_columnconfigure(1, weight=1, minsize=300)
        root.grid_rowconfigure(0, weight=1)

        self.status = tk.Label(side, text="", font=self.font_status, bg=BG, fg=INK, anchor="w", justify="left")
        self.status.pack(fill="x")
        self.info = tk.Label(side, text="", font=self.font, bg=BG, fg=MUTED, anchor="w", justify="left")
        self.info.pack(fill="x", pady=(4, 6))
        self.pass_button = ttk.Button(side, text="パス", command=self.on_pass)
        self.pass_button.pack(fill="x")

        tk.Label(side, text="棋譜", font=self.font_bold, bg=BG, fg=MUTED, anchor="w").pack(fill="x", pady=(12, 2))
        log_frame = tk.Frame(side, bg=BG)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, width=34, height=12, font=self.font, wrap="word", relief="flat",
                           bg="#ffffff", fg=INK, padx=8, pady=6, state="disabled", highlightthickness=0)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.log.tag_configure("you", foreground=YOU, font=self.font_bold)
        self.log.tag_configure("agent", foreground=AGENT, font=self.font_bold)
        self.log.tag_configure("muted", foreground=MUTED)

        box = tk.LabelFrame(side, text=" 新しい対局 ", font=self.font_bold, bg=BG, fg=MUTED, padx=8, pady=6)
        box.pack(fill="x", pady=(12, 0))
        tk.Label(box, text="相手 (match.py と同じ指定)", font=self.font_small, bg=BG, fg=MUTED,
                 anchor="w").grid(row=0, column=0, columnspan=4, sticky="w")
        self.agent_var = tk.StringVar(value=agent_spec)
        ttk.Combobox(box, textvariable=self.agent_var, values=list(AGENTS), font=self.font).grid(
            row=1, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        tk.Label(box, text="先手", font=self.font_small, bg=BG, fg=MUTED).grid(row=2, column=0, sticky="w")
        self.first_var = tk.StringVar(value=FIRST_LABELS[first])
        ttk.Combobox(box, textvariable=self.first_var, values=list(FIRST_LABELS.values()), state="readonly",
                     width=8, font=self.font).grid(row=2, column=1, sticky="w", padx=(4, 12))
        tk.Label(box, text="シード", font=self.font_small, bg=BG, fg=MUTED).grid(row=2, column=2, sticky="w")
        self.seed_var = tk.StringVar(value="" if seed is None else str(seed))
        ttk.Entry(box, textvariable=self.seed_var, width=10, font=self.font).grid(row=2, column=3, sticky="ew",
                                                                                  padx=(4, 0))
        box.grid_columnconfigure(3, weight=1)
        ttk.Button(box, text="対局を始める", command=self.on_new_game).grid(row=3, column=0, columnspan=4,
                                                                         sticky="ew", pady=(8, 0))
        self.message = tk.Label(side, text="", font=self.font_small, bg=BG, fg=AGENT, anchor="w", justify="left",
                                wraplength=300)
        self.message.pack(fill="x", pady=(6, 0))

        self.canvas.tag_bind("hand", "<Button-1>", self.on_hand_click)
        self.canvas.tag_bind("target", "<Button-1>", self.on_column_click)
        for tag in ("hand", "target"):
            self.canvas.tag_bind(tag, "<Enter>", lambda _e: self.canvas.configure(cursor="hand2"))
            self.canvas.tag_bind(tag, "<Leave>", lambda _e: self.canvas.configure(cursor=""))
        root.bind("<Key>", self.on_key)
        root.report_callback_exception = self.report_error
        root.after(100, self.poll)

        self.start(agent_spec, first, seed)

    def px(self, size: float, *style) -> tuple:
        """文字の大きさをピクセルで指定したフォント (拡大率を掛ける)"""
        return (self.family, -round(size * self.scale)) + style

    # --- 対局の進行 -----------------------------------------------------------

    def start(self, agent_spec: str, first: str, seed: int | None) -> bool:
        self.message.configure(text=f"{agent_spec} を読み込んでいます...")
        self.root.update_idletasks()
        try:
            game = Game(agent_spec, first, seed)
        except ValueError as e:
            self.message.configure(text=str(e))
            return False
        self.message.configure(text="")
        self.game_id += 1
        self.game = game
        self.selected = None
        self.thinking_since = None
        self.root.title(f"バトルライン — あなた vs {game.label}")
        self.advance()
        return True

    def advance(self) -> None:
        """あなたの自動パスを済ませ, エージェントの手番なら別のスレッドで考えさせる"""
        game = self.game
        while game.auto_pass():
            pass
        if game.agent_turn:
            self.thinking_since = time.perf_counter()
            threading.Thread(target=self._think, args=(game, self.game_id), daemon=True).start()
        self.redraw()
        if game.state.is_terminal:
            game_id = self.game_id
            self.root.after(300, lambda: game_id == self.game_id and self.messagebox.showinfo(
                "対局終了", game.result_text(), parent=self.root))

    def _think(self, game: Game, game_id: int) -> None:
        try:
            move, elapsed = game.think()
            self.results.put((game_id, move, elapsed, None))
        except Exception:
            self.results.put((game_id, None, None, traceback.format_exc()))

    def report_error(self, exc, value, tb) -> None:
        """画面の処理で起きた例外を, コンソールと画面の両方に出す (黙って操作を受け付けなくなるのを防ぐ)"""
        text = "".join(traceback.format_exception(exc, value, tb))
        print(text, flush=True)
        self.message.configure(text=f"エラー: {value!r} (詳細はコンソール)")

    def poll(self) -> None:
        """エージェントの手を受け取る (tkinter は別のスレッドから触れないので, ここで盤面を進める)"""
        try:
            self._poll()
        except Exception:
            self.report_error(*sys.exc_info())
        finally:                        # 例外が起きても受け取りを続ける (止まるとエージェントの手が反映されない)
            self.root.after(100, self.poll)

    def _poll(self) -> None:
        try:
            while True:
                game_id, move, elapsed, error = self.results.get_nowait()
                if game_id != self.game_id:            # 考えている間に新しい対局が始まった
                    continue
                self.thinking_since = None
                if error is None:
                    try:
                        self.game.play(move, "agent", elapsed)
                    except ValueError:
                        error = f"{self.game.label} が非合法手を返しました: {move!r}"
                if error is not None:
                    self.game.error = error
                    self.message.configure(text=error.strip().splitlines()[-1])
                    print(error, flush=True)
                    self.redraw()
                    continue
                self.advance()
        except queue.Empty:
            pass
        if self.thinking_since is not None:
            self.update_status()

    def play(self, move: int) -> None:
        if not self.game.my_turn:
            return
        self.selected = None
        self.game.play(move, "you")
        self.advance()

    # --- 入力 ------------------------------------------------------------------

    def target_flags(self) -> set:
        """選んでいるカードを置けるフラッグ"""
        if self.selected is None:
            return set()
        return {int(pybl.move_flag(m)) for m in self.game.legal_moves()
                if m != pybl.PASS_MOVE and int(pybl.move_card(m)) == self.selected}

    def _tag_value(self, prefix: str):
        for tag in self.canvas.gettags("current"):
            if tag.startswith(prefix):
                return int(tag[len(prefix):])
        return None

    def on_hand_click(self, _event) -> None:
        card = self._tag_value("card:")
        if card is None or not self.game.my_turn:
            return
        self.selected = None if self.selected == card else card
        self.redraw()

    def on_column_click(self, _event) -> None:
        flag = self._tag_value("col:")
        if flag is not None:
            self.place(flag)

    def place(self, flag: int) -> None:
        if self.selected is not None and flag in self.target_flags():
            self.play(int(pybl.make_move(self.selected, flag)))

    def on_key(self, event) -> None:
        if isinstance(event.widget, (self.tk.Entry, self.ttk.Entry)):      # ttk.Combobox は ttk.Entry の子
            return
        if event.char and event.char in "123456789":
            self.place(int(event.char) - 1)
        elif event.keysym == "Escape" and self.selected is not None:
            self.selected = None
            self.redraw()

    def on_pass(self) -> None:
        if int(pybl.PASS_MOVE) in self.game.legal_moves():
            self.play(int(pybl.PASS_MOVE))

    def on_new_game(self) -> None:
        text = self.seed_var.get().strip()
        try:
            seed = int(text) if text else None
        except ValueError:
            self.message.configure(text="シードは整数で指定してください")
            return
        first = {label: key for key, label in FIRST_LABELS.items()}[self.first_var.get()]
        self.start(self.agent_var.get().strip() or DEFAULT_AGENT, first, seed)

    # --- 描画 ------------------------------------------------------------------

    def update_status(self) -> None:
        game = self.game
        if game.error:
            text = "エラーで止まりました"
        elif game.state.is_terminal:
            text = game.result_text().splitlines()[0]
        elif game.my_turn:
            text = "あなたの番です" if self.selected is None else "置くフラッグを選んでください"
        else:
            elapsed = 0.0 if self.thinking_since is None else time.perf_counter() - self.thinking_since
            text = f"{game.label} が考えています… {elapsed:.1f} 秒"
        self.status.configure(text=text, fg=AGENT if game.agent_turn else INK)

    def redraw(self) -> None:
        game = self.game
        state = game.state
        # get_hand は numpy.uint64 を返す. NumPy 1.x では uint64 と int のシフトが TypeError になるので int にする
        if self.selected is not None and (not game.my_turn or not int(state.get_hand(game.me)) >> self.selected & 1):
            self.selected = None
        self.update_status()
        claimed_you = bin(state.get_claimed_flags(game.me)).count("1")
        claimed_agent = bin(state.get_claimed_flags(game.opp)).count("1")
        self.info.configure(text=f"{game.turn} 手目　山札 {state.deck_count} 枚\n"
                                 f"確保　あなた {claimed_you} - {claimed_agent} {game.label}\n"
                                 f"先手　{'あなた' if state.first_player == game.me else game.label}")
        self.pass_button.configure(state="normal" if int(pybl.PASS_MOVE) in game.legal_moves() else "disabled")
        self.draw_board()
        self.draw_log()

    def draw_board(self) -> None:
        c = self.canvas
        game = self.game
        state = game.state
        c.delete("all")
        targets = self.target_flags()
        last = None
        if game.last_move and game.last_move[0] == "agent" and game.last_move[1] != pybl.PASS_MOVE:
            last = int(pybl.move_card(game.last_move[1]))

        # 相手の手札 (終局後は表を見せる)
        c.create_text(MARGIN, MARGIN + 14, text=f"{game.label} の陣地", anchor="w", font=self.font_bold, fill=AGENT)
        if state.is_terminal:
            cards = sorted_cards(state.get_hand(game.opp))
            x = BOARD_W - MARGIN - len(cards) * 40
            c.create_text(x - 8, MARGIN + 14, text="手札:" if cards else "手札: なし", anchor="e",
                          font=self.font_small, fill=MUTED)
            for k, card in enumerate(cards):
                self.draw_card(x + k * 40, MARGIN, 36, 30, int(card), font_size=14)
        else:
            count = int(state.get_hand_count(game.opp))
            x = BOARD_W - MARGIN - count * 22
            c.create_text(x - 8, MARGIN + 14, text="手札", anchor="e", font=self.font_small, fill=MUTED)
            for k in range(count):
                self.round_rect(x + k * 22, MARGIN + 1, x + k * 22 + 18, MARGIN + 29, 3, fill="#6d6559",
                                outline="#8a8274")

        for i in range(pybl.NUM_FLAGS):
            x0 = MARGIN + i * (COL_W + COL_GAP)
            x1 = x0 + COL_W
            is_target = i in targets
            tags = ("col", f"col:{i}") + (("target",) if is_target else ())
            self.round_rect(x0, BOARD_TOP, x1, BOARD_BOTTOM, 10, fill=FELT_TARGET if is_target else FELT,
                            outline=ACCENT if is_target else FELT, width=3 if is_target else 1, tags=tags)
            owner = state.get_flag_owner(i)
            for who, top, direction in (("agent", FLAG_TOP - CARD_GAP - CARD_H, -1), ("you", MY_TOP, 1)):
                player = game.me if who == "you" else game.opp
                cards = state.get_flag_cards(i, player)
                for k in range(3):
                    y = top + direction * k * (CARD_H + CARD_GAP)
                    if k < len(cards):
                        card = int(cards[k])
                        self.draw_card(x0 + 6, y, COL_W - 12, CARD_H, card, font_size=20, highlight=card == last,
                                       tags=tags)
                    else:
                        self.round_rect(x0 + 6, y, x1 - 6, y + CARD_H, 6, fill=SLOT, outline="", tags=tags)
                if who == "agent":
                    label_y = top - 2 * (CARD_H + CARD_GAP) - 17
                else:
                    label_y = top + 2 * (CARD_H + CARD_GAP) + CARD_H + 17
                c.create_text((x0 + x1) / 2, label_y, text=game.formation(i, who).replace(" ", "\n合計 "),
                              tags=tags, justify="center", fill=INK if owner == player else MUTED,
                              font=self.px(10, *(("bold",) if owner == player else ())))
            # フラッグ (確保した側に寄せて, その色で塗る)
            cx, cy, r = (x0 + x1) / 2, FLAG_TOP + FLAG_H / 2, 17
            if owner == game.me:
                fill, ink, line, cy = YOU, "#ffffff", YOU, cy + 9
            elif owner == game.opp:
                fill, ink, line, cy = AGENT, "#ffffff", AGENT, cy - 9
            else:
                fill, ink, line = "#ffffff", MUTED, SLOT
            c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline=line, width=2, tags=tags)
            c.create_text(cx, cy, text=str(i + 1), font=self.font_bold, fill=ink, tags=tags)

        # あなたの手札
        c.create_text(MARGIN, BOARD_BOTTOM + 17, text="あなた の陣地 / 手札", anchor="w", font=self.font_bold, fill=YOU)
        hint = ""
        if game.my_turn:
            hint = ("カードをクリックして選ぶ" if self.selected is None
                    else "置くフラッグの列をクリック (数字キーでも可、Esc で取り消し)")
        c.create_text(BOARD_W - MARGIN, BOARD_BOTTOM + 17, text=hint, anchor="e", font=self.font_small, fill=MUTED)
        hand = [int(card) for card in sorted_cards(state.get_hand(game.me))]
        playable = {int(pybl.move_card(m)) for m in game.legal_moves() if m != pybl.PASS_MOVE}
        total = len(hand) * HAND_W + max(len(hand) - 1, 0) * 10
        x = (BOARD_W - total) / 2
        for card in hand:
            lift = 12 if card == self.selected else 0
            tags = ("hand", f"card:{card}") if card in playable else ()
            self.draw_card(x, HAND_TOP + 12 - lift, HAND_W, HAND_H - 12, card, font_size=30,
                           dim=game.my_turn and card not in playable, selected=card == self.selected, tags=tags)
            x += HAND_W + 10
        if self.scale != 1.0:
            c.scale("all", 0, 0, self.scale, self.scale)

    def draw_card(self, x, y, w, h, card: int, font_size=16, highlight=False, dim=False, selected=False, tags=()):
        text = pybl.card_to_str(card)
        color, rank = text[0], text[1:]
        fill = CARD_COLORS[color]
        ink = CARD_INK.get(color, "#ffffff")
        if dim:
            fill = self._blend(fill, BG, 0.55)
        outline, width = "#ffffff", 1
        if selected:
            outline, width = ACCENT, 4
        elif highlight:
            outline, width = LAST, 4
        self.round_rect(x, y, x + w, y + h, 6, fill=fill, outline=outline, width=width, tags=tags)
        self.canvas.create_text(x + 6, y + 3, text=color, anchor="nw", font=self.px(max(9, font_size // 2)),
                                fill=ink, tags=tags)
        self.canvas.create_text(x + w / 2, y + h / 2 + 2, text=rank, font=self.px(font_size, "bold"),
                                fill=ink, tags=tags)

    def round_rect(self, x0, y0, x1, y1, r, **kwargs):
        points = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
                  x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        return self.canvas.create_polygon(points, smooth=True, **kwargs)

    @staticmethod
    def _blend(color: str, other: str, t: float) -> str:
        a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
        b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(p + (q - p) * t):02x}" for p, q in zip(a, b))

    def draw_log(self) -> None:
        game = self.game
        log = self.log
        log.configure(state="normal")
        log.delete("1.0", "end")
        for turn, who, move, elapsed, note, claims in game.log:
            log.insert("end", f"{turn:>3}  ", "muted")
            log.insert("end", "あなた" if who == "you" else game.label, who)
            log.insert("end", f": {move}")
            if elapsed is not None:
                log.insert("end", f"  ({elapsed:.2f} 秒)", "muted")
            if note:
                log.insert("end", f"  {note}", "muted")
            log.insert("end", "\n")
            for flag, owner in claims:
                log.insert("end", f"       → フラッグ {flag + 1} を")
                log.insert("end", "あなた" if owner == "you" else game.label, owner)
                log.insert("end", "が確保\n")
        if game.state.is_terminal:
            log.insert("end", "\n" + game.result_text() + "\n", "you" if game.state.winner == game.me else "agent")
        log.configure(state="disabled")
        log.see("end")


def main():
    parser = argparse.ArgumentParser(description="あなたと指定したエージェントでバトルラインを GUI で対戦する")
    parser.add_argument("agent", nargs="?", default=DEFAULT_AGENT,
                        help=f"対戦するエージェント ({', '.join(AGENTS)}, または モジュール名:関数名. "
                             f"@引数名=値 で引数を指定. 省略時は {DEFAULT_AGENT})")
    parser.add_argument("--first", choices=FIRST_CHOICES, default="random", help="先手 (既定はランダム)")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード (配札とエージェントの乱数を固定する)")
    args = parser.parse_args()

    try:
        import tkinter as tk
    except ImportError:
        raise SystemExit("tkinter が使えません. tkinter 付きの Python (python.org の配布版など) で実行してください")
    if sys.platform == "win32":
        try:                            # 高 DPI の画面で文字がぼやけないようにする
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    load_agent(args.agent)              # ウィンドウを開く前に, 指定の誤りを知らせる (誤りなら SystemExit)
    root = tk.Tk()
    App(root, args.agent, args.first, args.seed)
    root.mainloop()


if __name__ == "__main__":
    main()
