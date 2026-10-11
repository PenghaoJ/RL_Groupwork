import subprocess
import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


FILES = "abcdefghi"
ENGINE_PATH = Path(__file__).resolve().parent / "engine" / "Pikafish-Windows-x86-64-universal.exe"
START_FEN = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r"
PIECE_TEXT = {
    "K": "\u5e05",
    "A": "\u4ed5",
    "E": "\u76f8",
    "H": "\u9a6c",
    "R": "\u8f66",
    "C": "\u70ae",
    "P": "\u5175",
    "k": "\u5c06",
    "a": "\u58eb",
    "e": "\u8c61",
    "h": "\u9a6c",
    "r": "\u8f66",
    "c": "\u70ae",
    "p": "\u5352",
}
PIECE_FILL = {
    True: "#cf3f2e",
    False: "#222222",
}


def parse_fen(fen: str):
    board = {}
    board_part, side = fen.split()
    rows = board_part.split("/")
    for fen_y, row in enumerate(rows):
        y = 9 - fen_y
        x = 0
        for ch in row:
            if ch.isdigit():
                x += int(ch)
                continue
            board[(x, y)] = ch
            x += 1
    return board, side


def square_name(pos):
    x, y = pos
    return f"{FILES[x]}{y}"


def move_name(src, dst):
    return f"{square_name(src)}{square_name(dst)}"


def side_name(is_red: bool):
    return "Red" if is_red else "Black"


class Engine:
    def __init__(self, engine_path: str):
        engine_path = Path(engine_path).resolve()
        self.process = subprocess.Popen(
            [str(engine_path)],
            cwd=str(engine_path.parent),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.send("uci")
        self._wait_for("uciok")
        self.send("setoption name Threads value 16")
        self.send("setoption name Hash value 512")
        self.send("isready")
        self._wait_for("readyok")

    def send(self, text: str):
        if not self.process.stdin:
            raise RuntimeError("Engine stdin is unavailable.")
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

    def _wait_for(self, target: str):
        if not self.process.stdout:
            raise RuntimeError("Engine stdout is unavailable.")
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Engine exited unexpectedly.")
            if line.strip() == target:
                return

    def bestmove(self, moves, depth=12):
        position = "position startpos"
        if moves:
            position += " moves " + " ".join(moves)
        self.send(position)
        self.send(f"go depth {depth}")
        if not self.process.stdout:
            raise RuntimeError("Engine stdout is unavailable.")
        info_lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Engine exited unexpectedly.")
            line = line.strip()
            if line.startswith("info "):
                info_lines.append(line)
            if line.startswith("bestmove "):
                return line.split()[1], info_lines[-8:]

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        self.process.terminate()


class XiangqiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Chinese Chess Board")
        self.root.geometry("900x680")

        self.board, self.side = parse_fen(START_FEN)
        self.moves = []
        self.selected = None
        self.history = []
        self.legal_targets = []
        self.last_move = None
        self.last_move_capture = False
        self.engine = Engine(ENGINE_PATH)
        self.auto_play = False
        self.auto_job = None
        self.auto_delay_ms = 700
        self.side_to_move_is_red = True
        self.game_result = None
        self.result_popup_shown = False
        self.position_counts = {}
        self.halfmove_clock = 0
        self.engine_depth = tk.IntVar(value=12)
        self.red_human = tk.BooleanVar(value=True)
        self.black_human = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar()
        self.log_var = tk.StringVar()
        self.auto_var = tk.StringVar()

        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(root, width=540, height=600, bg="#f4d7a1", highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=12, pady=12, sticky="nw")

        panel = ttk.Frame(root, padding=(0, 12, 12, 12))
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(7, weight=1)

        ttk.Label(panel, text="Chinese Chess Board", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        message_box = ttk.LabelFrame(panel, text="Status", padding=10)
        message_box.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        message_box.grid_propagate(False)
        message_box.configure(height=170)
        message_box.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(
            message_box,
            textvariable=self.status_var,
            justify="left",
            wraplength=290,
            font=("Segoe UI", 10, "bold"),
        )
        self.status_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.log_label = ttk.Label(
            message_box,
            textvariable=self.log_var,
            justify="left",
            wraplength=290,
        )
        self.log_label.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self.auto_label = ttk.Label(
            message_box,
            textvariable=self.auto_var,
            justify="left",
            wraplength=290,
        )
        self.auto_label.grid(row=2, column=0, sticky="ew")

        mode_box = ttk.LabelFrame(panel, text="Side Mode")
        mode_box.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        mode_box.columnconfigure(0, weight=1)

        ttk.Checkbutton(mode_box, text="Red is Human", variable=self.red_human, command=self.on_mode_change).grid(
            row=0, column=0, sticky="w", padx=8, pady=6
        )
        ttk.Checkbutton(mode_box, text="Black is Human", variable=self.black_human, command=self.on_mode_change).grid(
            row=1, column=0, sticky="w", padx=8, pady=(0, 8)
        )

        button_box = ttk.Frame(panel)
        button_box.grid(row=3, column=0, sticky="ew")
        button_box.columnconfigure(0, weight=1)

        buttons = [
            ("Engine Move", self.engine_move),
            ("Undo Full Move", self.undo_move),
            ("New Game", self.reset_board),
            ("Start Auto-play", self.start_autoplay),
            ("Stop Auto-play", self.stop_autoplay),
        ]
        for row_index, (label, command) in enumerate(buttons):
            ttk.Button(button_box, text=label, command=command).grid(row=row_index, column=0, sticky="ew", pady=4)

        depth_row = ttk.Frame(panel)
        depth_row.grid(row=4, column=0, sticky="ew", pady=(10, 8))
        depth_row.columnconfigure(1, weight=1)
        ttk.Label(depth_row, text="Engine Depth").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(depth_row, from_=1, to=30, textvariable=self.engine_depth, width=6).grid(
            row=0, column=1, sticky="e"
        )

        ttk.Label(panel, text="Move List", font=("Segoe UI", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(2, 6))

        moves_frame = ttk.Frame(panel)
        moves_frame.grid(row=7, column=0, sticky="nsew")
        moves_frame.columnconfigure(0, weight=1)
        moves_frame.rowconfigure(0, weight=1)

        self.moves_box = tk.Text(moves_frame, width=34, height=16, state="disabled", font=("Consolas", 10))
        self.moves_box.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(moves_frame, orient="vertical", command=self.moves_box.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.moves_box.configure(yscrollcommand=scrollbar.set)

        self.canvas.bind("<Button-1>", self.on_click)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.update_mode_text()
        self.set_status("Red to move. Click a piece, then click a target square.")
        self.set_log("Pikafish is connected.")
        self.draw_board()
        self.record_position()
        self.maybe_schedule_engine_turn()

    def set_status(self, text: str):
        self.status_var.set(text)

    def set_log(self, text: str):
        self.log_var.set(text[:140])

    def update_mode_text(self):
        red_mode = "Human" if self.red_human.get() else "Engine"
        black_mode = "Human" if self.black_human.get() else "Engine"
        auto_text = "Auto-play is on." if self.auto_play else "Auto-play is off."
        self.auto_var.set(f"{auto_text} Red: {red_mode}. Black: {black_mode}.")

    def current_side_is_human(self):
        return self.red_human.get() if self.side_to_move_is_red else self.black_human.get()

    def position_key(self):
        pieces = sorted((x, y, piece) for (x, y), piece in self.board.items())
        return (tuple(pieces), self.side_to_move_is_red)

    def record_position(self):
        key = self.position_key()
        self.position_counts[key] = self.position_counts.get(key, 0) + 1

    def get_winner(self):
        pieces = set(self.board.values())
        red_alive = "K" in pieces
        black_alive = "k" in pieces
        if red_alive and black_alive:
            return None
        if red_alive:
            return "Red"
        if black_alive:
            return "Black"
        return "Draw"

    def update_game_result(self):
        winner = self.get_winner()
        self.game_result = winner
        if not winner:
            key = self.position_key()
            if self.position_counts.get(key, 0) >= 3:
                winner = "Draw"
                self.game_result = winner
                self.set_status("Game over. Draw.")
                self.set_log("Threefold repetition.")
            elif self.halfmove_clock >= 120:
                winner = "Draw"
                self.game_result = winner
                self.set_status("Game over. Draw.")
                self.set_log("No capture for 120 half-moves.")
        if winner == "Draw":
            if "Draw" not in self.status_var.get():
                self.set_status("Game over. Draw.")
            if not self.log_var.get():
                self.set_log("Draw by rule.")
            self.show_result_popup("Draw")
        elif winner:
            self.set_status(f"Game over. {winner} wins.")
            self.set_log(f"{winner} captured the opposing king.")
            self.show_result_popup(winner)
        return winner

    def show_result_popup(self, result):
        if self.result_popup_shown:
            return
        self.result_popup_shown = True
        if result == "Draw":
            messagebox.showinfo("Game Over", "Draw")
        else:
            messagebox.showinfo("Game Over", f"{result} wins")

    def draw_board(self):
        self.canvas.delete("all")
        margin = 30
        cell = 60
        for i in range(9):
            x = margin + i * cell
            self.canvas.create_line(x, margin, x, margin + 4 * cell, width=2, fill="#694f2f")
            self.canvas.create_line(x, margin + 5 * cell, x, margin + 9 * cell, width=2, fill="#694f2f")
        for i in range(10):
            y = margin + i * cell
            self.canvas.create_line(margin, y, margin + 8 * cell, y, width=2, fill="#694f2f")
        self.canvas.create_line(margin + 3 * cell, margin, margin + 5 * cell, margin + 2 * cell, fill="#694f2f", width=2)
        self.canvas.create_line(margin + 5 * cell, margin, margin + 3 * cell, margin + 2 * cell, fill="#694f2f", width=2)
        self.canvas.create_line(
            margin + 3 * cell, margin + 7 * cell, margin + 5 * cell, margin + 9 * cell, fill="#694f2f", width=2
        )
        self.canvas.create_line(
            margin + 5 * cell, margin + 7 * cell, margin + 3 * cell, margin + 9 * cell, fill="#694f2f", width=2
        )
        self.canvas.create_text(270, 300, text="CHINESE CHESS", font=("Segoe UI", 18, "bold"), fill="#8b2a19")

        if self.last_move:
            self.draw_last_move_arrow(self.last_move[0], self.last_move[1])

        for x, y in self.legal_targets:
            cx = margin + x * cell
            cy = margin + (9 - y) * cell
            self.canvas.create_oval(
                cx - 9,
                cy - 9,
                cx + 9,
                cy + 9,
                fill="#4a90e2",
                outline="",
            )

        for (x, y), piece in self.board.items():
            cx = margin + x * cell
            cy = margin + (9 - y) * cell
            selected = self.selected == (x, y)
            self.canvas.create_oval(
                cx - 24,
                cy - 24,
                cx + 24,
                cy + 24,
                fill="#f8efdc",
                outline="#d59a4d" if selected else "#7f6240",
                width=4 if selected else 2,
            )
            self.canvas.create_text(
                cx,
                cy,
                text=PIECE_TEXT[piece],
                font=("KaiTi", 22, "bold"),
                fill=PIECE_FILL[piece.isupper()],
            )

    def draw_last_move_arrow(self, src, dst):
        margin = 30
        cell = 60
        x1 = margin + src[0] * cell
        y1 = margin + (9 - src[1]) * cell
        x2 = margin + dst[0] * cell
        y2 = margin + (9 - dst[1]) * cell
        self.canvas.create_oval(x1 - 20, y1 - 20, x1 + 20, y1 + 20, outline="#2f7de1", width=3)
        self.canvas.create_oval(x2 - 22, y2 - 22, x2 + 22, y2 + 22, outline="#e14b2f", width=4)
        self.canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill="#2f7de1",
            width=4,
            arrow=tk.LAST,
            arrowshape=(14, 16, 6),
            smooth=True,
        )
        self.canvas.create_oval(x1 - 6, y1 - 6, x1 + 6, y1 + 6, fill="#2f7de1", outline="")
        if self.game_result:
            self.canvas.create_oval(x2 - 10, y2 - 10, x2 + 10, y2 + 10, fill="#e14b2f", outline="")

    def on_mode_change(self):
        self.update_mode_text()
        self.maybe_schedule_engine_turn()

    def on_click(self, event):
        if self.game_result:
            self.set_status(f"Game over. {self.game_result} wins." if self.game_result != "Draw" else "Game over. Draw.")
            return
        if self.auto_play:
            self.set_status("Stop auto-play before making a manual move.")
            return
        if not self.current_side_is_human():
            self.set_status(f"{side_name(self.side_to_move_is_red)} is set to Engine.")
            self.maybe_schedule_engine_turn()
            return
        pos = self.pixel_to_pos(event.x, event.y)
        if pos is None:
            return
        piece = self.board.get(pos)
        expected_red = self.side_to_move_is_red
        if self.selected is None:
            if not piece or piece.isupper() != expected_red:
                self.set_status(f"{side_name(expected_red)} to move. Select one of that side's pieces.")
                return
            self.selected = pos
            self.legal_targets = self.get_legal_targets(pos)
            self.set_status(f"Selected {square_name(pos)}. Click a destination square.")
            self.draw_board()
            return

        src = self.selected
        self.selected = None
        self.legal_targets = []
        if src == pos:
            self.set_status("Selection cleared.")
            self.draw_board()
            return
        if self.try_user_move(src, pos):
            self.draw_board()
            self.maybe_schedule_engine_turn()
        else:
            self.set_status("That move is not legal. Try again.")
            self.draw_board()

    def pixel_to_pos(self, x, y):
        margin = 30
        cell = 60
        board_x = round((x - margin) / cell)
        board_y = round((y - margin) / cell)
        if not (0 <= board_x <= 8 and 0 <= board_y <= 9):
            return None
        px = margin + board_x * cell
        py = margin + board_y * cell
        if abs(px - x) > 24 or abs(py - y) > 24:
            return None
        return board_x, 9 - board_y

    def try_user_move(self, src, dst):
        piece = self.board.get(src)
        if not piece or piece.isupper() != self.side_to_move_is_red:
            return False
        if not self.is_legal_move(piece, src, dst):
            return False
        self.push_move(src, dst)
        winner = self.update_game_result()
        if not winner:
            self.set_status(f"You played {move_name(src, dst)}. {side_name(self.side_to_move_is_red)} to move.")
            self.set_log("Ready for the next move.")
        return True

    def push_move(self, src, dst):
        captured = self.board.get(dst)
        moving_piece = self.board[src]
        self.history.append((src, dst, moving_piece, captured, self.side_to_move_is_red))
        self.board[dst] = moving_piece
        del self.board[src]
        notation = move_name(src, dst)
        self.moves.append(notation)
        self.side_to_move_is_red = not self.side_to_move_is_red
        self.last_move = (src, dst)
        self.last_move_capture = captured is not None
        self.halfmove_clock = 0 if captured else self.halfmove_clock + 1
        self.record_position()
        self.append_move_log(notation)

    def pop_move(self):
        if not self.history:
            return False
        src, dst, piece, captured, side_before_move = self.history.pop()
        self.board[src] = piece
        if captured:
            self.board[dst] = captured
        else:
            self.board.pop(dst, None)
        if self.moves:
            self.moves.pop()
        self.side_to_move_is_red = side_before_move
        self.last_move = None
        self.last_move_capture = False
        if self.history:
            prev_src, prev_dst, _, _, _ = self.history[-1]
            self.last_move = (prev_src, prev_dst)
        self.rebuild_position_state()
        self.refresh_moves_box()
        return True

    def rebuild_position_state(self):
        self.position_counts = {}
        temp_board, _ = parse_fen(START_FEN)
        temp_side_to_move_is_red = True
        self.position_counts[(tuple(sorted((x, y, piece) for (x, y), piece in temp_board.items())), temp_side_to_move_is_red)] = 1
        self.halfmove_clock = 0
        for src, dst, piece, captured, side_before_move in self.history:
            temp_board[dst] = temp_board[src]
            del temp_board[src]
            temp_side_to_move_is_red = not side_before_move
            key = (tuple(sorted((x, y, p) for (x, y), p in temp_board.items())), temp_side_to_move_is_red)
            self.position_counts[key] = self.position_counts.get(key, 0) + 1
            self.halfmove_clock = 0 if captured else self.halfmove_clock + 1

    def get_legal_targets(self, src):
        piece = self.board.get(src)
        if not piece:
            return []
        targets = []
        for x in range(9):
            for y in range(10):
                dst = (x, y)
                if self.is_legal_move(piece, src, dst):
                    targets.append(dst)
        return targets

    def append_move_log(self, notation):
        self.moves_box.configure(state="normal")
        self.moves_box.insert("end", notation + "\n")
        self.moves_box.see("end")
        self.moves_box.configure(state="disabled")

    def refresh_moves_box(self):
        self.moves_box.configure(state="normal")
        self.moves_box.delete("1.0", "end")
        for move in self.moves:
            self.moves_box.insert("end", move + "\n")
        self.moves_box.configure(state="disabled")

    def engine_move(self):
        if self.game_result:
            self.set_status(f"Game over. {self.game_result} wins." if self.game_result != "Draw" else "Game over. Draw.")
            return
        if self.current_side_is_human():
            self.set_status(f"{side_name(self.side_to_move_is_red)} is set to Human.")
            return
        self.run_engine_move()

    def run_engine_move(self):
        try:
            bestmove, info = self.engine.bestmove(self.moves, depth=self.engine_depth.get())
        except Exception as exc:
            self.stop_autoplay()
            messagebox.showerror("Pikafish Error", str(exc))
            return False
        src = (FILES.index(bestmove[0]), int(bestmove[1]))
        dst = (FILES.index(bestmove[2]), int(bestmove[3]))
        piece = self.board.get(src)
        if not piece:
            self.stop_autoplay()
            self.set_status(f"Engine returned {bestmove}, but the board does not match.")
            return False
        if piece.isupper() != self.side_to_move_is_red:
            self.stop_autoplay()
            self.set_status(f"Engine returned {bestmove} for the wrong side.")
            return False
        self.push_move(src, dst)
        winner = self.update_game_result()
        if not winner:
            self.set_status(f"Engine played {bestmove}. {side_name(self.side_to_move_is_red)} to move.")
            self.set_log(info[-1] if info else f"bestmove {bestmove}")
        else:
            self.stop_autoplay()
        self.draw_board()
        return True

    def start_autoplay(self):
        if self.auto_play:
            return
        self.auto_play = True
        self.update_mode_text()
        self.set_status("Auto-play started.")
        self.maybe_schedule_engine_turn()

    def maybe_schedule_engine_turn(self):
        self.update_mode_text()
        if self.auto_job is not None:
            self.root.after_cancel(self.auto_job)
            self.auto_job = None
        if not self.auto_play:
            return
        if self.update_game_result():
            self.stop_autoplay()
            return
        if self.current_side_is_human():
            self.set_status(f"{side_name(self.side_to_move_is_red)} is waiting for a human move.")
            return
        self.auto_job = self.root.after(self.auto_delay_ms, self.autoplay_step)

    def autoplay_step(self):
        self.auto_job = None
        if not self.auto_play or self.current_side_is_human():
            return
        if self.update_game_result():
            self.stop_autoplay()
            return
        if self.run_engine_move():
            self.maybe_schedule_engine_turn()

    def stop_autoplay(self):
        self.auto_play = False
        if self.auto_job is not None:
            self.root.after_cancel(self.auto_job)
            self.auto_job = None
        self.update_mode_text()

    def undo_move(self):
        self.stop_autoplay()
        undone = False
        if self.pop_move():
            undone = True
        if self.pop_move():
            undone = True
        if undone:
            self.selected = None
            self.legal_targets = []
            self.game_result = None
            self.result_popup_shown = False
            self.set_status(f"Undid one full move. {side_name(self.side_to_move_is_red)} to move.")
            self.set_log("You can continue from here.")
            self.draw_board()

    def reset_board(self):
        self.stop_autoplay()
        self.board, self.side = parse_fen(START_FEN)
        self.moves.clear()
        self.history.clear()
        self.selected = None
        self.legal_targets = []
        self.last_move = None
        self.last_move_capture = False
        self.side_to_move_is_red = True
        self.game_result = None
        self.result_popup_shown = False
        self.position_counts = {}
        self.halfmove_clock = 0
        self.set_status("Red to move. Click a piece, then click a target square.")
        self.set_log("Board reset.")
        self.refresh_moves_box()
        self.draw_board()
        self.record_position()
        self.maybe_schedule_engine_turn()

    def is_game_over(self):
        return self.get_winner() is not None

    def on_close(self):
        self.stop_autoplay()
        self.engine.close()
        self.root.destroy()

    def is_legal_move(self, piece, src, dst):
        if src == dst:
            return False
        target = self.board.get(dst)
        if target and target.isupper() == piece.isupper():
            return False
        x1, y1 = src
        x2, y2 = dst
        dx = x2 - x1
        dy = y2 - y1
        adx = abs(dx)
        ady = abs(dy)

        def clear_path():
            if x1 == x2:
                step = 1 if y2 > y1 else -1
                return sum((x1, y) in self.board for y in range(y1 + step, y2, step))
            if y1 == y2:
                step = 1 if x2 > x1 else -1
                return sum((x, y1) in self.board for x in range(x1 + step, x2, step))
            return -1

        lower = piece.lower()
        if lower == "r":
            return (x1 == x2 or y1 == y2) and clear_path() == 0
        if lower == "c":
            blockers = clear_path()
            if blockers < 0:
                return False
            if target:
                return blockers == 1
            return blockers == 0
        if lower == "h":
            if (adx, ady) not in {(1, 2), (2, 1)}:
                return False
            leg = (x1 + (dx // 2 if adx == 2 else 0), y1 + (dy // 2 if ady == 2 else 0))
            return leg not in self.board
        if lower == "e":
            if adx != 2 or ady != 2:
                return False
            if piece.isupper() and y2 > 4:
                return False
            if piece.islower() and y2 < 5:
                return False
            eye = (x1 + dx // 2, y1 + dy // 2)
            return eye not in self.board
        if lower == "a":
            if adx != 1 or ady != 1:
                return False
            if x2 not in {3, 4, 5}:
                return False
            return (0 <= y2 <= 2) if piece.isupper() else (7 <= y2 <= 9)
        if lower == "k":
            if x2 not in {3, 4, 5}:
                return False
            palace_ok = (0 <= y2 <= 2) if piece.isupper() else (7 <= y2 <= 9)
            if adx + ady == 1 and palace_ok:
                return True
            if x1 == x2:
                step = 1 if y2 > y1 else -1
                pieces_between = sum((x1, y) in self.board for y in range(y1 + step, y2, step))
                return target and target.lower() == "k" and pieces_between == 0
            return False
        if lower == "p":
            forward = 1 if piece.isupper() else -1
            crossed = y1 >= 5 if piece.isupper() else y1 <= 4
            if dy == forward and dx == 0:
                return True
            if crossed and dy == 0 and adx == 1:
                return True
            return False
        return False


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    XiangqiApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        messagebox.showerror("Launch Failed", str(exc))
        sys.exit(1)
