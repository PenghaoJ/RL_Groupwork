"""A small, dependency-free Xiangqi MCTS policy using UCB1.

Values are from the player-to-move perspective at each node. Backpropagation
negates the value at every edge, as required for an alternating zero-sum game.
"""
from collections import Counter
import json
import math
import random
import sys
import time
from pathlib import Path

# The model worker adds this model folder to sys.path; add the app folder too
# so this policy always uses the project's authoritative rule implementation.
APP_DIR = Path(__file__).resolve().parents[2]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from rules import PRESETS, color, coords, legal_moves, make_fen, other, parse_fen  # noqa: E402

PIECE_VALUE = {"k": 0, "a": 2, "e": 2, "h": 4, "c": 4.5, "r": 9, "p": 1}
EXPLORATION = math.sqrt(2.0)
EXPERIENCE_FILE = Path(__file__).resolve().with_name("experience.json")


def _position_key(board, side):
    return " ".join(make_fen(board, side).split()[:2])


def _material(board, perspective):
    score = 0.0
    for row in board:
        for piece in row:
            if piece:
                value = PIECE_VALUE[piece.lower()]
                score += value if color(piece) == perspective else -value
    return score


def _terminal_value(board, side, legal, repetitions, halfmove, ply, max_plies):
    """Return (finished, value-for-side-to-move), matching Game outcomes."""
    if not legal:
        return True, -1.0
    key = _position_key(board, side)
    if repetitions.get(key, 0) >= 3 or halfmove >= 120 or ply >= max_plies:
        return True, 0.0
    return False, 0.0


def _apply(board, move):
    r, c = coords(move[:2])
    tr, tc = coords(move[2:])
    captured = board[tr][tc]
    board[tr][tc], board[r][c] = board[r][c], ""
    return bool(captured)


class Node:
    __slots__ = ("board", "side", "halfmove", "ply", "repetitions", "moves",
                 "experience", "children", "visits", "value_sum", "move", "captured")

    def __init__(self, board, side, halfmove, ply, repetitions, moves,
                 move=None, captured=False, experience=None):
        self.board = board
        self.side = side
        self.halfmove = halfmove
        self.ply = ply
        self.repetitions = repetitions
        self.moves = moves
        self.experience = experience if experience is not None else {}
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.move = move
        self.captured = captured

    @property
    def mean(self):
        return self.value_sum / self.visits if self.visits else 0.0


def _child(parent, move):
    board = [row[:] for row in parent.board]
    captured = _apply(board, move)
    side = other(parent.side)
    halfmove = 0 if captured else parent.halfmove + 1
    ply = parent.ply + 1
    repetitions = parent.repetitions.copy()
    key = _position_key(board, side)
    repetitions[key] += 1
    moves = legal_moves(board, side)
    node = Node(board, side, halfmove, ply, repetitions, moves, move, captured,
                parent.experience)
    parent.children[move] = node
    return node


def _select_child(node):
    log_parent = math.log(max(1, node.visits))
    return max(
        node.children.values(),
        key=lambda child: _combined_value(node, child)
        + EXPLORATION * math.sqrt(log_parent / child.visits),
    )


def _experience_key(board, side, move):
    return _position_key(board, side) + "|" + move


def _combined_value(parent, child):
    record = parent.experience.get(_experience_key(parent.board, parent.side, child.move), {})
    visits = max(0, int(record.get("visits", 0)))
    if not visits:
        return -child.mean
    prior_visits = min(200, visits)
    prior_value = float(record.get("value_sum", 0.0)) / visits
    return (child.visits * -child.mean + prior_visits * prior_value) / (child.visits + prior_visits)


def _rollout(node, max_plies, rng, depth=10):
    board = [row[:] for row in node.board]
    side, halfmove, ply = node.side, node.halfmove, node.ply
    repetitions = node.repetitions.copy()
    for _ in range(depth):
        moves = legal_moves(board, side)
        done, value = _terminal_value(board, side, moves, repetitions, halfmove, ply, max_plies)
        if done:
            return value
        # Uniform random legal play gives a simple rollout policy. Small tactical
        # preference for captures reduces obviously wasteful simulations.
        captures = []
        for move in moves:
            tr, tc = coords(move[2:])
            if board[tr][tc]:
                captures.append(move)
        move = rng.choice(captures) if captures and rng.random() < 0.65 else rng.choice(moves)
        captured = _apply(board, move)
        halfmove = 0 if captured else halfmove + 1
        side = other(side)
        ply += 1
        repetitions[_position_key(board, side)] += 1
    score = _material(board, side)
    # Bounded heuristic value keeps UCB rewards in a stable range.
    return math.tanh(score / 8.0) * 0.8


class Policy:
    def __init__(self, checkpoint=None, device="cpu"):
        self.rng = random.Random()
        self.iterations = 96
        self.rollout_depth = 10
        self.experience_path = EXPERIENCE_FILE
        self._experience_mtime = None
        self.experience = {}
        self.scene_id = None
        self._last_ply = None

    def _identify_scene(self, observation):
        # Starting position takes precedence. Keep its identity as pieces disappear.
        key = " ".join(observation["fen"].split()[:2])
        starts = [" ".join(p["fen"].split()[:2]) for p in PRESETS[:4]]
        ply = observation.get("ply", 0)
        new_game = self._last_ply is None or ply == 0 or ply < self._last_ply
        if key in starts and (new_game or self.scene_id is None):
            scene = starts.index(key)
            if scene != self.scene_id:
                self._experience_mtime = None
            self.scene_id = scene
        elif self._last_ply is not None and ply < self._last_ply:
            self.scene_id = None
            self._experience_mtime = None
        self._last_ply = ply

    def _load_experience(self):
        try:
            mtime = self.experience_path.stat().st_mtime_ns
        except OSError:
            self.experience, self._experience_mtime = {}, None
            return
        cache_key = (mtime, self.scene_id)
        if cache_key == self._experience_mtime:
            return
        data = json.loads(self.experience_path.read_text(encoding="utf-8"))
        self.experience = {}
        if data.get("memory_routing") == "isolated-123-to-4.v1":
            if self.scene_id in (0, 1, 2, 3):
                sources = [self.scene_id] if self.scene_id < 3 else [0, 1, 2, 3]
                for source in sources:
                    for key, record in data["scenes"][str(source)]["values"].items():
                        item = self.experience.setdefault(key, {"visits": 0, "value_sum": 0.0})
                        item["visits"] += record["visits"]
                        item["value_sum"] += record["value_sum"]
            # Unknown/custom starting boards use search only, never mixed memory.
        # Old unseparated values have unknown provenance: do not load them.
        self._experience_mtime = cache_key

    def select_move(self, observation, time_budget_ms=800):
        move, _ = self.select_move_with_pi(observation, time_budget_ms)
        return move

    def select_move_with_pi(self, observation, time_budget_ms=800):
        self._identify_scene(observation)
        legal = observation["legal_moves"]
        if not legal:
            raise ValueError("No legal moves in the supplied observation.")
        if len(legal) == 1:
            return legal[0], {legal[0]: 1.0}

        board, side, halfmove, _ = parse_fen(observation["fen"])
        self._load_experience()
        repetitions = Counter(observation.get("repetition_counts", {}))
        root = Node(board, side, halfmove, observation.get("ply", 0),
                    repetitions, legal, experience=self.experience)
        # Keep a modest reserve for request serialization and avoid exceeding the
        # host's advisory time budget. Always perform at least a few simulations.
        budget = max(25, int(time_budget_ms or 800))
        deadline = time.monotonic() + max(0.05, budget / 1000.0 * 0.8)
        completed = 0
        while completed < self.iterations and (completed < 4 or time.monotonic() < deadline):
            node = root
            path = [root]
            while node.moves and len(node.children) == len(node.moves):
                node = _select_child(node)
                path.append(node)
                done, _ = _terminal_value(node.board, node.side, node.moves,
                                          node.repetitions, node.halfmove,
                                          node.ply, root_max_plies(observation))
                if done:
                    break
            if node.moves and len(node.children) < len(node.moves):
                unexpanded = [move for move in node.moves if move not in node.children]
                node = _child(node, self.rng.choice(unexpanded))
                path.append(node)
            value = _rollout(node, root_max_plies(observation), self.rng, self.rollout_depth)
            for visited in reversed(path):
                visited.visits += 1
                visited.value_sum += value
                value = -value
            completed += 1

        if not root.children:
            return self.rng.choice(legal), {move: 1.0 / len(legal) for move in legal}
        # Most visits is robust to noisy rollout values; UCB visits every root move
        # before choosing, so the root move remains legal by construction.
        visits = {move: root.children[move].visits for move in root.children}
        visit_total = sum(visits.values())
        pi = {move: visits.get(move, 0) / visit_total for move in legal}
        move = max(root.children.values(), key=lambda child: (
            child.visits,
            _combined_value(root, child),
        )).move
        return move, pi


def root_max_plies(observation):
    return int(observation.get("max_plies", 160))
