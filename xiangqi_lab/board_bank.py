"""Separate training and validation starts with a reproducible balanced schedule."""
import copy
import hashlib
import json
import random

from rules import Game, make_fen


def position_key(fen):
    return " ".join(Game(fen).fen.split()[:2])


def mirror_key(fen):
    game = Game(fen)
    return " ".join(make_fen([list(reversed(row)) for row in game.board], game.side).split()[:2])


class BoardBank:
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        if data.get("schema") != "xiangqi-lab.board-bank.v1":
            raise ValueError("Unknown board bank schema.")
        self.training = self.data["training"]
        self.validation = self.data["validation"]
        if not self.training or not self.validation:
            raise ValueError("Training and validation boards are both required.")
        forbidden = set()
        for fen in self.data.get("excluded_fens", []):
            forbidden.update([position_key(fen), mirror_key(fen)])
        seen, names = set(), set()
        for item in self.training + self.validation:
            game = Game(item["fen"])
            key, mirrored = position_key(item["fen"]), mirror_key(item["fen"])
            if game.side != "red" or game.outcome() is not None:
                raise ValueError("Use active Red-to-move starts.")
            if item["id"] in names or key in seen or key in forbidden:
                raise ValueError("Duplicate, mirrored, overlapping or excluded board.")
            names.add(item["id"])
            seen.update([key, mirrored])
        self.sha256 = hashlib.sha256(json.dumps(self.data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def training_position(self, episode, seed):
        if episode < 1:
            raise ValueError("Episode numbers start at one.")
        cycle, index = divmod(episode - 1, len(self.training))
        order = list(range(len(self.training)))
        random.Random(f"{self.sha256}:train-order:{seed}:{cycle}").shuffle(order)
        return self.training[order[index]]
