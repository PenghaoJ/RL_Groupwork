"""Dependency-free Xiangqi rules and a reusable, turn-based experiment state.

Coordinates: board[row][col], row 0 is Black's home rank; UCI a0 is Red's.
Research draw policy: threefold position repetition / 120 non-capture plies.
This is not the full competition perpetual-check/chase adjudicator.
"""
from collections import Counter
import copy
import re

START_FEN = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1"
PIECES = "KAEHRCPkaehrcp"
PRESETS = [
    {"id": "rook", "name": "Rook & pawn", "tag": "Starter · 6 pieces", "description": "Coordinate a rook and an advancing pawn.", "fen": "4k4/4a4/9/9/4p4/2P6/9/4R4/9/3K5 w - - 0 1"},
    {"id": "cannon", "name": "Cannon coordination", "tag": "Explore · 8 pieces", "description": "Explore cannon screens and pawn coordination.", "fen": "3aka3/9/9/4p4/9/4P4/2C6/9/9/3AK4 w - - 0 1"},
    {"id": "horse", "name": "Horse & pawn", "tag": "Explore · 8 pieces", "description": "Control horse paths and key squares.", "fen": "3ak4/9/4e4/9/4p4/2P6/5H3/9/9/3AK4 w - - 0 1"},
    {"id": "start", "name": "Starting position", "tag": "Reference · 32 pieces", "description": "A full board for demonstrations and rule checks.", "fen": START_FEN},
]

def color(piece):
    return "red" if piece.isupper() else "black"

def other(side):
    return "black" if side == "red" else "red"

def square(row, col):
    return chr(97 + col) + str(9 - row)

def coords(name):
    if not re.fullmatch(r"[a-i][0-9]", name):
        raise ValueError("Coordinates must be between a0 and i9.")
    return 9 - int(name[1]), ord(name[0]) - 97

def action_id(move):
    r1, c1 = coords(move[:2]); r2, c2 = coords(move[2:])
    return (r1 * 9 + c1) * 90 + r2 * 9 + c2

def action_move(action):
    if type(action) is not int or not 0 <= action < 8100:
        raise ValueError("Action IDs must be between 0 and 8099.")
    src, dst = divmod(action, 90)
    return square(*divmod(src, 9)) + square(*divmod(dst, 9))

def inside(r, c):
    return 0 <= r < 10 and 0 <= c < 9

def palace(r, c, side):
    return 3 <= c <= 5 and (7 <= r <= 9 if side == "red" else 0 <= r <= 2)

def pseudo(board, r, c, tr, tc):
    piece = board[r][c]
    if not piece or (r, c) == (tr, tc) or not inside(tr, tc):
        return False
    target = board[tr][tc]
    side = color(piece)
    if target and color(target) == side:
        return False
    dr, dc = tr-r, tc-c
    ar, ac = abs(dr), abs(dc)
    kind = piece.lower()
    blockers = -1
    if r == tr:
        blockers = sum(bool(board[r][i]) for i in range(min(c, tc)+1, max(c, tc)))
    elif c == tc:
        blockers = sum(bool(board[i][c]) for i in range(min(r, tr)+1, max(r, tr)))
    if kind == "r":
        return blockers == 0
    if kind == "c":
        return blockers == (1 if target else 0)
    if kind == "h":
        if (ar, ac) == (2, 1):
            return not board[r + dr//2][c]
        if (ar, ac) == (1, 2):
            return not board[r][c + dc//2]
        return False
    if kind == "e":
        return ar == ac == 2 and (tr >= 5 if side == "red" else tr <= 4) and not board[r+dr//2][c+dc//2]
    if kind == "a":
        return ar == ac == 1 and palace(tr, tc, side)
    if kind == "k":
        return (ar+ac == 1 and palace(tr, tc, side)) or (target and target.lower() == "k" and c == tc and blockers == 0)
    if kind == "p":
        return (dr == (-1 if side == "red" else 1) and dc == 0) or (dr == 0 and ac == 1 and (r <= 4 if side == "red" else r >= 5))
    return False

def in_check(board, side):
    king = "K" if side == "red" else "k"
    pos = next(((r,c) for r in range(10) for c in range(9) if board[r][c] == king), None)
    if pos is None:
        return True
    return any(p and color(p) != side and pseudo(board,r,c,*pos) for r,row in enumerate(board) for c,p in enumerate(row))

def legal_moves(board, side):
    moves = []
    for r,row in enumerate(board):
        for c,p in enumerate(row):
            if not p or color(p) != side:
                continue
            for tr in range(10):
                for tc in range(9):
                    target = board[tr][tc]
                    if target.lower() == "k" or not pseudo(board,r,c,tr,tc):
                        continue
                    board[r][c], board[tr][tc] = "", p
                    safe = not in_check(board,side)
                    board[r][c], board[tr][tc] = p, target
                    if safe:
                        moves.append(square(r,c)+square(tr,tc))
    return moves

def parse_fen(fen):
    if not isinstance(fen,str) or len(fen) > 250:
        raise ValueError("Invalid FEN format.")
    fields = fen.strip().split()
    if len(fields) not in (2,6) or fields[1] not in ("w","r","b"):
        raise ValueError("FEN requires a board and side w/r/b, optionally followed by - - halfmove fullmove.")
    rows = fields[0].replace("n","h").replace("N","H").replace("b","e").replace("B","E").split("/")
    if len(rows) != 10:
        raise ValueError("FEN must contain 10 ranks.")
    board = []
    for row in rows:
        result = []
        for ch in row:
            if ch in "123456789": result.extend([""] * int(ch))
            elif ch in PIECES: result.append(ch)
            else: raise ValueError("FEN contains an unknown piece.")
        if len(result) != 9: raise ValueError("Each FEN rank must contain 9 squares.")
        board.append(result)
    side = "black" if fields[1] == "b" else "red"
    half, full = 0, 1
    if len(fields) == 6:
        if fields[2:4] != ["-","-"]: raise ValueError("FEN fields 3 and 4 must be - -.")
        try: half, full = int(fields[4]), int(fields[5])
        except ValueError: raise ValueError("FEN move counters must be integers.")
        if not 0 <= half <= 100000 or not 1 <= full <= 100000: raise ValueError("Move counters are out of range.")
    counts = Counter(p for row in board for p in row if p)
    for p,limit in zip("KAEHRCP",[1,2,2,2,2,2,5]):
        if counts[p] > limit or counts[p.lower()] > limit: raise ValueError("Piece counts exceed standard Xiangqi limits.")
    if counts["K"] != 1 or counts["k"] != 1: raise ValueError("Each side must have exactly one king.")
    for r,row in enumerate(board):
        for c,p in enumerate(row):
            if not p: continue
            s = color(p)
            if p.lower() in "ka" and not palace(r,c,s): raise ValueError("Kings and advisors must be inside their own palace.")
            if p.lower() == "a" and (r,c) not in ([(9,3),(9,5),(8,4),(7,3),(7,5)] if s == "red" else [(0,3),(0,5),(1,4),(2,3),(2,5)]): raise ValueError("An advisor is on an invalid square.")
            if p.lower() == "e" and (r,c) not in ([(9,2),(9,6),(7,0),(7,4),(7,8),(5,2),(5,6)] if s == "red" else [(0,2),(0,6),(2,0),(2,4),(2,8),(4,2),(4,6)]): raise ValueError("An elephant is on an invalid square.")
            if p.lower() == "p" and ((s == "red" and (r > 6 or (r >= 5 and c%2))) or (s == "black" and (r < 3 or (r <= 4 and c%2)))): raise ValueError("A pawn is on an unreachable square.")
    if in_check(board, other(side)):
        raise ValueError("The non-moving side is in check or the kings face each other. Adjust the position or side to move.")
    return board, side, half, full

def make_fen(board, side, half=0, full=1):
    rows=[]
    for row in board:
        out=""; empty=0
        for p in row:
            if not p: empty+=1
            else:
                if empty: out+=str(empty); empty=0
                out+=p
        if empty: out+=str(empty)
        rows.append(out)
    return "/".join(rows)+f" {'w' if side == 'red' else 'b'} - - {half} {full}"

class Game:
    def __init__(self, fen=START_FEN, max_plies=160):
        self.board, self.side, self.halfmove, self.fullmove = parse_fen(fen)
        self.initial_fen = self.fen
        self.max_plies = max_plies
        self.history=[]
        self.repetitions=Counter({self.key:1})
        self._legal=None

    @property
    def fen(self): return make_fen(self.board,self.side,self.halfmove,self.fullmove)
    @property
    def key(self): return " ".join(self.fen.split()[:2])
    @property
    def legal(self):
        if self._legal is None: self._legal=legal_moves(self.board,self.side)
        return self._legal

    def outcome(self):
        if not self.legal:
            return {"winner":other(self.side), "reason":"checkmate" if in_check(self.board,self.side) else "stalemate", "terminated":True, "truncated":False}
        if self.repetitions[self.key] >= 3: reason="repetition"
        elif self.halfmove >= 120: reason="no_capture"
        elif len(self.history) >= self.max_plies:
            return {"winner":None,"reason":"ply_limit","terminated":False,"truncated":True}
        else: return None
        return {"winner":None,"reason":reason,"terminated":True,"truncated":False}

    def step(self, move, actor="human", analysis=None):
        if self.outcome(): raise ValueError("The episode has ended. Reset or load another position.")
        if move not in self.legal: raise ValueError("Illegal move: you cannot leave your king in check or expose facing kings.")
        r,c=coords(move[:2]); tr,tc=coords(move[2:])
        entry={"move":move,"side":self.side,"piece":self.board[r][c],"captured":self.board[tr][tc],"fen_before":self.fen,"actor":actor,"analysis":analysis or {}}
        self.board[tr][tc], self.board[r][c] = self.board[r][c], ""
        self.halfmove=0 if entry["captured"] else self.halfmove+1
        if self.side == "black": self.fullmove+=1
        self.side=other(self.side); self._legal=None
        self.repetitions[self.key]+=1
        entry["fen_after"]=self.fen
        self.history.append(entry)
        return self.outcome()

    def undo(self):
        if not self.history: return
        self.repetitions[self.key]-=1
        entry=self.history.pop()
        self.board,self.side,self.halfmove,self.fullmove=parse_fen(entry["fen_before"])
        self._legal=None

    def observation(self):
        planes=[[[int(p == kind) for p in row] for row in self.board] for kind in PIECES]
        return {"fen":self.fen,"board_planes":planes,"plane_order":PIECES,"side_to_move":self.side,"legal_moves":list(self.legal),"legal_action_ids":[action_id(m) for m in self.legal],"halfmove_clock":self.halfmove,"ply":len(self.history),"max_plies":self.max_plies,"repetition_counts":dict(self.repetitions)}

    def snapshot(self):
        return {"fen":self.fen,"initial_fen":self.initial_fen,"board":copy.deepcopy(self.board),"side":self.side,"legal_moves":list(self.legal),"in_check":in_check(self.board,self.side),"history":copy.deepcopy(self.history),"outcome":self.outcome(),"halfmove_clock":self.halfmove,"max_plies":self.max_plies,"piece_count":sum(bool(p) for row in self.board for p in row)}

    def export(self):
        return {"schema":"xiangqi-lab.episode.v1","rules":"xiangqi-research-v1","draw_policy":"threefold-position-or-120-noncapture-plies","initial_fen":self.initial_fen,"max_plies":self.max_plies,"moves":copy.deepcopy(self.history),"final_fen":self.fen,"outcome":self.outcome()}
