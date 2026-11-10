"""play_gui.py (tkinter で対戦する GUI) のテスト. 画面のテストは tkinter と画面が使えるときだけ行う"""
import time

import pytest

import pybl
from play_gui import Game, agent_label


def play_out(game: Game) -> None:
    """あなたは先頭の合法手を, エージェントはその場で考えて指し, 終局させる"""
    while not game.state.is_terminal:
        if game.auto_pass():
            continue
        if game.my_turn:
            game.play(game.legal_moves()[0], "you")
        else:
            move, elapsed = game.think()
            game.play(move, "agent", elapsed)


@pytest.mark.parametrize("first", ["you", "agent"])
def test_play_to_end(first):
    game = Game("random", first, seed=3)
    assert (game.state.first_player == game.me) == (first == "you")
    assert game.my_turn == (first == "you") and game.agent_turn == (first == "agent")
    play_out(game)
    assert game.state.is_terminal and not game.my_turn and not game.agent_turn
    assert len(game.log) == game.state.move_count
    assert game.result_text()


def test_same_seed_same_game():
    logs = []
    for _ in range(2):
        game = Game("random", "random", seed=11)
        play_out(game)
        logs.append([(who, move) for _, who, move, *_ in game.log])
    assert logs[0] == logs[1]


def test_legal_moves_and_errors():
    game = Game("random", "you", seed=2)
    assert game.legal_moves() == [int(m) for m in game.state.observe(game.me).get_legal_moves()]
    with pytest.raises(ValueError):
        game.play(pybl.PASS_MOVE, "you")            # 置ける場所があるのでパスはできない
    with pytest.raises(ValueError):
        game.play(game.legal_moves()[0], "agent")   # エージェントの手番ではない
    game.play(game.legal_moves()[0], "you")
    assert game.legal_moves() == []                 # あなたの手番ではない
    assert game.agent_turn


def test_think_does_not_change_state():
    game = Game("random", "agent", seed=4)
    before = game.state.move_count
    move, elapsed = game.think()
    assert game.state.move_count == before and game.state.is_legal(move) and elapsed >= 0


def test_invalid_spec():
    with pytest.raises(ValueError):
        Game("no_such_agent")
    with pytest.raises(ValueError):
        Game("random", "nobody")


def test_formation_and_claims():
    game = Game("random", "you", seed=5)
    play_out(game)
    state = game.state
    for i in range(pybl.NUM_FLAGS):
        for who, player in (("you", game.me), ("agent", game.opp)):
            assert bool(game.formation(i, who)) == (len(state.get_flag_cards(i, player)) == 3)
    claims = [owner for *_, cs in game.log for _, owner in cs]
    assert claims.count("you") == bin(state.get_claimed_flags(game.me)).count("1")
    assert claims.count("agent") == bin(state.get_claimed_flags(game.opp)).count("1")


def test_agent_label():
    assert agent_label("abdeck") == "abdeck"
    assert agent_label("abdeck@depth=3") == "abdeck"
    assert agent_label("az@checkpoint=runs/az6/checkpoints/gen_0021.pt,device=cuda") == "az (gen_0021)"


@pytest.fixture
def root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("画面が使えない")
    root.withdraw()
    yield root
    root.destroy()


def wait_until(root, condition, timeout=20.0):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline
        root.update()
        time.sleep(0.01)


def click(app, item) -> None:
    """canvas の item をクリックしたことにする"""
    x0, y0, x1, y1 = app.canvas.bbox(item)
    x, y = (x0 + x1) // 2, (y0 + y1) // 2
    app.canvas.event_generate("<Motion>", x=x, y=y)
    app.canvas.event_generate("<Button-1>", x=x, y=y)
    app.canvas.event_generate("<ButtonRelease-1>", x=x, y=y)


def test_app_plays_game(root, monkeypatch):
    from play_gui import App
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *a, **k: None)
    app = App(root, "random", "agent", 7)
    root.update()
    while not app.game.state.is_terminal:
        wait_until(root, lambda: app.game.my_turn or app.game.state.is_terminal)
        if app.game.state.is_terminal:
            break
        if app.game.legal_moves() == [int(pybl.PASS_MOVE)]:
            continue
        move = next(m for m in app.game.legal_moves() if m != pybl.PASS_MOVE)
        card, flag = int(pybl.move_card(move)), int(pybl.move_flag(move))
        app.selected = card                     # 手札のクリックは下の test で確かめる
        app.redraw()
        assert flag in app.target_flags()
        app.place(flag)
    assert app.game.error is None
    assert len(app.game.log) == app.game.state.move_count


def test_app_click_and_new_game(root, monkeypatch):
    from play_gui import App
    monkeypatch.setattr("tkinter.messagebox.showinfo", lambda *a, **k: None)
    root.deiconify()
    app = App(root, "random", "you", 1)
    root.update()
    hand = app.canvas.find_withtag("hand")
    assert hand
    card = int(next(t for t in app.canvas.gettags(hand[0]) if t.startswith("card:"))[5:])
    click(app, hand[0])
    root.update()
    assert app.selected == card
    targets = app.canvas.find_withtag("target")
    assert targets
    flag = int(next(t for t in app.canvas.gettags(targets[0]) if t.startswith("col:"))[4:])
    click(app, targets[0])
    root.update()
    assert app.game.log[0][1:3] == ("you", pybl.move_to_str(pybl.make_move(card, flag)))
    wait_until(root, lambda: app.game.my_turn)
    old_id = app.game_id
    app.seed_var.set("abc")
    app.on_new_game()
    assert app.game_id == old_id and "整数" in app.message.cget("text")
    app.seed_var.set("3")
    app.first_var.set("相手")
    app.on_new_game()
    assert app.game_id == old_id + 1
    wait_until(root, lambda: app.game.my_turn)
    assert app.game.log[0][1] == "agent"
