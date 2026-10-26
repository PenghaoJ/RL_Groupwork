"""Check sampling odds, weighted updates, uniform compatibility and resume."""
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
import torch
from dqn_model import QNetwork, encode_state
from dqn_training import ENDGAMES, TrainConfig, export_policy, optimize, sample_replay, train, transition_group
from rules import Game


def example_replay():
    state = encode_state(Game(ENDGAMES["rook"]["fen"]).observation())
    # Ten live records, one win, two losses and three terminal draws.
    return [(state, i, reward, state, terminal, () if terminal else (0, 1))
            for i, (reward, terminal) in enumerate([(0., False)] * 10 + [(1., True)] +
                                                    [(-1., True)] * 2 + [(0., True)] * 3)]


class TerminalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_group_probabilities_and_unbiased_weights(self):
        replay = example_replay()
        config = TrainConfig(replay_sampling="terminal_mixture", batch_size=64)
        original = copy.deepcopy(replay)
        counts = [0] * len(replay)
        rng = random.Random(42)
        seen_weights = {}
        for _ in range(300):
            batch, weights, info = sample_replay(replay, rng, config)
            self.assertTrue(info["mixture_active"])
            for item, weight in zip(batch, weights):
                i = item[1]
                counts[i] += 1
                seen_weights[i] = weight
        expected = [0.5 / 16 + (0 if i < 10 else 0.5 / (3 * (1 if i == 10 else 2 if i < 13 else 3)))
                    for i in range(16)]
        self.assertAlmostEqual(sum(expected), 1.)
        self.assertEqual(replay, original)
        for i, probability in enumerate(expected):
            self.assertAlmostEqual(seen_weights[i], 1 / (16 * probability))
            self.assertLess(abs(counts[i] / sum(counts) - probability), 0.012)
        # Correct any per-record loss, not only the terminal reward average.
        self.assertAlmostEqual(sum(p * seen_weights[i] * i**2 for i, p in enumerate(expected)),
                               sum(i**2 for i in range(16)) / 16)
        self.assertEqual(transition_group(replay[0]), "live")

    def test_empty_groups_and_uniform_fallback_preserve_rng(self):
        replay = example_replay()
        for items, mode, alpha in ((replay[:10], "terminal_mixture", .5),
                                    (replay, "uniform", .5), (replay, "terminal_mixture", 0.)):
            first, second = random.Random(5), random.Random(5)
            batch, weights, info = sample_replay(items, first,
                TrainConfig(batch_size=4, replay_sampling=mode, terminal_mix=alpha))
            self.assertEqual(batch, second.sample(items, 4))
            self.assertEqual(weights, [1.] * 4)
            self.assertEqual(first.getstate(), second.getstate())
            self.assertFalse(info["mixture_active"])
        items = replay[:11]  # Only the win group exists.
        batch, weights, _ = sample_replay(items, random.Random(6),
            TrainConfig(batch_size=64, replay_sampling="terminal_mixture"))
        for item, weight in zip(batch, weights):
            self.assertAlmostEqual(weight, 1 / (11 * (.5 / 11 + (.5 if item[4] else 0))))

    def test_weighted_huber_gradient(self):
        class ConstantQ(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Parameter(torch.zeros(8100))
            def forward(self, batch):
                return self.q[None, :].expand(len(batch[0]), -1)
        replay = example_replay()
        for algorithm in ("dqn", "double_dqn"):
            config = TrainConfig(algorithm=algorithm, batch_size=64, replay_sampling="terminal_mixture")
            model, target = ConstantQ(), ConstantQ()
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            batch, weights, _ = sample_replay(replay, random.Random(99), config)
            expected_gradient = torch.zeros(8100)
            expected_loss = 0.
            for item, weight in zip(batch, weights):
                expected_gradient[item[1]] -= item[2] * weight / 64
                expected_loss += .5 * item[2]**2 * weight / 64
            stats = {}
            loss = optimize(model, target, optimizer, replay, random.Random(99), config, "cpu", stats)
            self.assertAlmostEqual(loss, expected_loss, places=7)
            torch.testing.assert_close(model.q, -.1 * expected_gradient)
            self.assertEqual(stats["batches"], 1)
            self.assertEqual(stats["mixture_batches"], 1)
            self.assertEqual(sum(stats["sampled_" + k] for k in ("win", "loss", "draw", "live")), 64)

    def test_uniform_update_is_identical_to_frozen_baseline(self):
        baseline = ROOT / "output/dqn/ROOK20_TERMINAL50_20261025_S42/baselines/source/dqn_training.py"
        if not baseline.exists():
            self.skipTest("Frozen experiment snapshot is not included in the repository.")
        spec = importlib.util.spec_from_file_location("uniform_baseline_test", baseline)
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
        for algorithm in ("dqn", "double_dqn"):
            torch.manual_seed(42)
            first, target = QNetwork(), QNetwork()
            second = copy.deepcopy(first)
            left = torch.optim.Adam(first.parameters(), lr=.0003)
            right = torch.optim.Adam(second.parameters(), lr=.0003)
            rng1, rng2 = random.Random(43), random.Random(43)
            for _ in range(3):
                a = optimize(first, target, left, example_replay(), rng1,
                    TrainConfig(batch_size=8, algorithm=algorithm), "cpu")
                b = old.optimize(second, target, right, example_replay(), rng2,
                    old.TrainConfig(batch_size=8, algorithm=algorithm), "cpu")
                self.assertEqual(a, b)
                self.assertEqual(rng1.getstate(), rng2.getstate())
                self.assertTrue(all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items()))

    def test_terminal_resume_matches_uninterrupted(self):
        near_draw = {**ENDGAMES["rook"], "fen": ENDGAMES["rook"]["fen"].rsplit(" ", 2)[0] + " 119 1"}
        for algorithm in ("dqn", "double_dqn"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as temp, patch.dict(ENDGAMES, rook=near_draw):
                config = TrainConfig(algorithm=algorithm, min_episodes=2, max_episodes=4,
                    max_plies=4, eval_every=2, eval_games=2, patience=10, batch_size=2,
                    warmup_steps=2, replay_size=32, train_every=1, target_every=2,
                    device="cpu", replay_sampling="terminal_mixture")
                folder, full = Path(temp) / "resume", Path(temp) / "full"
                train(config, folder)
                train(replace(config, max_episodes=6), folder, resume=True)
                train(replace(config, max_episodes=6), full)
                a = torch.load(folder / "last.pt", weights_only=True)
                b = torch.load(full / "last.pt", weights_only=True)
                self.assertGreater(a["sampling_stats"]["mixture_batches"], 0)
                self.assertEqual(a["sampling_stats"], b["sampling_stats"])
                self.assertEqual(a["replay_rng"], b["replay_rng"])
                self.assertTrue(all(torch.equal(v, b["model"][k]) for k, v in a["model"].items()))
                with zipfile.ZipFile(export_policy(folder)) as package:
                    self.assertTrue(any(n.endswith("/sampling.json") for n in package.namelist()))

    def test_invalid_sampling_config(self):
        for value in (-.1, 1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                TrainConfig(terminal_mix=value).validate()
        with self.assertRaises(ValueError):
            TrainConfig(replay_sampling="unknown").validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
