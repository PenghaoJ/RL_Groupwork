"""Check split isolation, paired schedules, validation and checkpoint resume."""
import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
from board_bank import BoardBank
from rules import Game, make_fen
import torch
from dqn_model import QNetwork
from dqn_training import TrainConfig, evaluate, export_policy, train


class BoardBankTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "experiments/rook20_20261025/boards.json").read_text())
        self.bank = BoardBank(self.data)
        torch.set_num_threads(2)

    def test_generation_and_balanced_schedule(self):
        self.assertEqual((len(self.bank.training), len(self.bank.validation)), (20, 6))
        for item in self.bank.training + self.bank.validation:
            game = Game(item["generation"]["base_fen"])
            for move in item["generation"]["moves"]:
                game.step(move)
            self.assertEqual(make_fen(game.board, "red"), item["fen"])
        expected = {p["id"] for p in self.bank.training}
        for start in (1, 21, 1001):
            self.assertEqual({self.bank.training_position(e, 42)["id"] for e in range(start, start + 20)}, expected)
        clone = BoardBank(self.data)
        self.assertEqual([self.bank.training_position(e, 42) for e in range(1, 61)],
                         [clone.training_position(e, 42) for e in range(1, 61)])

    def test_rejects_overlap_mirror_and_old_test(self):
        for kind in ("overlap", "mirror", "test"):
            data = copy.deepcopy(self.data)
            if kind == "test":
                data["training"][0]["fen"] = data["excluded_fens"][0]
            else:
                game = Game(data["training"][0]["fen"])
                data["validation"][0]["fen"] = (game.fen if kind == "overlap" else
                    make_fen([list(reversed(row)) for row in game.board], "red"))
            with self.assertRaises(ValueError):
                BoardBank(data)

    def test_validation_is_balanced_frozen_and_reproducible(self):
        model = QNetwork()
        before = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        config = TrainConfig(max_plies=4, eval_games=12)
        first = evaluate(model, config, "cpu", positions=self.bank.validation)
        second = evaluate(model, config, "cpu", positions=self.bank.validation)
        self.assertEqual(first, second)
        self.assertEqual(set(first["per_board"]), {p["id"] for p in self.bank.validation})
        self.assertTrue(all(sum(row.values()) == 2 for row in first["per_board"].values()))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            evaluate(model, replace(config, eval_games=7), "cpu", positions=self.bank.validation)

    def test_training_resume_matches_uninterrupted_for_both_algorithms(self):
        for algorithm in ("dqn", "double_dqn"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as temp:
                config = TrainConfig(algorithm=algorithm, min_episodes=2, max_episodes=4,
                    max_plies=4, eval_every=2, eval_games=6, patience=5,
                    batch_size=4, warmup_steps=4, replay_size=32, train_every=1, target_every=2, device="cpu")
                folder = Path(temp) / "resume"
                calls = []
                train(config, folder, board_bank=self.bank, on_checkpoint=lambda path, improved: calls.append((path, improved)))
                self.assertEqual(len(calls), 2)
                train(replace(config, max_episodes=6), folder, resume=True, board_bank=self.bank)
                full = Path(temp) / "full"
                train(replace(config, max_episodes=6), full, board_bank=self.bank)
                resumed = torch.load(folder / "last.pt", weights_only=True)
                direct = torch.load(full / "last.pt", weights_only=True)
                self.assertEqual(resumed["board_bank_sha256"], self.bank.sha256)
                self.assertTrue(all(torch.equal(v, direct["model"][k]) for k, v in resumed["model"].items()))
                self.assertGreater(resumed["updates"], 0)
                with (folder / "episodes.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([r["board_id"] for r in rows], [self.bank.training_position(e, 42)["id"] for e in range(1, 7)])
                changed = copy.deepcopy(self.data)
                changed["id"] += "-changed"
                with self.assertRaisesRegex(ValueError, "board bank"):
                    train(replace(config, max_episodes=8), folder, resume=True, board_bank=BoardBank(changed))
                package = export_policy(folder, f"custom-{algorithm}-rook20-test", f"{algorithm} rook20 test")
                self.assertTrue(package.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
