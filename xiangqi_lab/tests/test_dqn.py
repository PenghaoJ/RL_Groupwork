"""Small deterministic training checks, never the 1,000-episode experiment."""
import copy
import csv
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from dqn_model import Policy, QNetwork, batch_states, encode_state
    from dqn_training import (ENDGAMES, EarlyStopping, FixedOpponentEnv, TrainConfig,
                              evaluate, export_policy, legal_next_values, train)
    from rules import Game


@unittest.skipUnless(HAS_TORCH, "Optional PyTorch training environment is not installed")
class DQNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_learner_win_and_loss_after_opponent_reply(self):
        def no_reply(*args):
            self.fail("Opponent replied after checkmate")
        env = FixedOpponentEnv(opponent=no_reply)
        env.reset("4k4/3R1R3/9/9/9/9/9/9/9/3K5 w - - 0 1")
        _, reward, ended, limited, info = env.step("d8e8")
        self.assertEqual((reward, ended, limited, info["environment_plies"]), (1, True, False, 1))
        env = FixedOpponentEnv(opponent=lambda obs, rng: "d1e1")
        env.reset("3k5/9/9/9/9/9/9/9/3r1r3/3K5 w - - 0 1")
        _, reward, ended, limited, info = env.step("d0e0")
        self.assertEqual((reward, ended, limited, info["environment_plies"]), (-1, True, False, 2))

    def test_black_opening_and_time_limit_end_on_learner_turn(self):
        for colour in ("red", "black"):
            env = FixedOpponentEnv(max_plies=4, learner=colour)
            obs, _ = env.reset(ENDGAMES["rook"]["fen"], seed=7)
            self.assertEqual(obs["side_to_move"], colour)
            self.assertEqual(len(env.game.history), int(colour == "black"))
            while env.game.outcome() is None:
                obs, reward, ended, limited, _ = env.step(obs["legal_action_ids"][0])
            self.assertTrue(limited)
            self.assertFalse(ended)
            self.assertEqual(obs["side_to_move"], colour)
            self.assertTrue(obs["legal_action_ids"])
            self.assertLessEqual(len(env.game.history), 4)

    def test_targets_mask_illegal_actions_and_bootstrap_truncation(self):
        state = encode_state(Game(ENDGAMES["rook"]["fen"]).observation())

        class FakeTarget(torch.nn.Module):
            def forward(self, batch):
                q = torch.zeros((len(batch[0]), 8100))
                q[:, 1], q[:, 2], q[:, 8000] = 3, 5, 1000
                return q

        values = legal_next_values(FakeTarget(), [state, state], [(1, 2), ()],
                                   [False, True], torch.device("cpu"))
        self.assertEqual(values.tolist(), [5, 0])

    def test_double_dqn_selects_online_but_uses_target_value(self):
        state = encode_state(Game(ENDGAMES["rook"]["fen"]).observation())

        class Scores(torch.nn.Module):
            def __init__(self, first, second):
                super().__init__()
                self.q = torch.nn.Parameter(torch.zeros(8100))
                with torch.no_grad():
                    self.q[1], self.q[2], self.q[8000] = first, second, 1000

            def forward(self, batch):
                return self.q.unsqueeze(0).expand(len(batch[0]), -1)

        online, target = Scores(10, 1), Scores(3, 5)
        device = torch.device("cpu")
        # The first transition may be time-limited, but is not terminal.
        args = (target, [state, state], [(1, 2), ()], [False, True], device)
        self.assertEqual(legal_next_values(*args).tolist(), [5, 0])
        double_values = legal_next_values(*args, online=online)
        self.assertEqual(double_values.tolist(), [3, 0])
        self.assertFalse(double_values.requires_grad)
        self.assertTrue(all(p.grad is None for net in (online, target) for p in net.parameters()))
        with self.assertRaises(ValueError):
            legal_next_values(target, [state], [()], [False], device, online=online)

    def test_history_is_retained_and_validation_does_not_train(self):
        game = Game(ENDGAMES["rook"]["fen"])
        game.step(game.legal[0])
        observation = game.observation()
        encoded = encode_state(observation)
        self.assertEqual(len(encoded[3]), len(game.repetitions))
        altered = copy.deepcopy(observation)
        old_key = next(key for key in altered["repetition_counts"] if key != game.key)
        altered["repetition_counts"][old_key] += 1
        self.assertNotEqual(encoded, encode_state(altered))
        model = QNetwork()
        before = copy.deepcopy(model.state_dict())
        rng_before = torch.get_rng_state().clone()
        config = TrainConfig(max_plies=4, eval_games=2)
        one = evaluate(model, config, torch.device("cpu"))
        two = evaluate(model, config, torch.device("cpu"))
        self.assertEqual(one, two)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_early_stop_waits_for_1000_then_full_patience(self):
        stopper = EarlyStopping(1000, 5, 0.02)
        for episode in range(100, 1000, 100):
            self.assertFalse(stopper.update(episode, 0.5)[1])
            self.assertEqual(stopper.bad_evals, 0)
        self.assertFalse(stopper.update(1000, 0.5)[1])
        for episode in (1100, 1200, 1300, 1400):
            self.assertFalse(stopper.update(episode, 0.5)[1])
        self.assertTrue(stopper.update(1500, 0.5)[1])
        self.assertFalse(stopper.update(1600, 0.6)[1])
        self.assertEqual(stopper.bad_evals, 0)
        with self.assertRaises(ValueError):
            stopper.update(1700, float("nan"))

    def test_short_train_resume_and_export_through_real_worker(self):
        for algorithm in ("dqn", "double_dqn"):
            with self.subTest(algorithm=algorithm):
                self.check_short_train_resume_export(algorithm)

    def check_short_train_resume_export(self, algorithm):
        config = TrainConfig(min_episodes=2, max_episodes=4, max_plies=8, eval_every=2,
                             eval_games=2, patience=2, batch_size=4, warmup_steps=4,
                             replay_size=32, train_every=1, target_every=2, device="cpu",
                             algorithm=algorithm)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "smoke"
            torch.manual_seed(config.seed)
            initial = QNetwork().state_dict()
            result = train(config, folder, source_hashes={"test": "smoke"})
            self.assertEqual(result["summary"]["episodes_completed"], 4)
            self.assertEqual(result["summary"]["algorithm"], algorithm)
            self.assertGreater(result["summary"]["updates"], 0)
            last = torch.load(folder / "last.pt", weights_only=True)
            self.assertEqual(last["config"]["algorithm"], algorithm)
            self.assertTrue(any(not torch.equal(initial[key], value) for key, value in last["model"].items()))
            self.assertEqual(last["total_plies"], 32)
            # An interrupted write after the checkpoint must not duplicate CSV rows on resume.
            with (folder / "episodes.csv").open("a", encoding="utf-8") as stream:
                stream.write("5,0,0,0,0,ply_limit,0.1,0,0\n")
            resumed = train(replace(config, max_episodes=6), folder, resume=True, source_hashes={"test": "smoke"})
            self.assertEqual(resumed["summary"]["episodes_completed"], 6)
            self.assertEqual(resumed["summary"]["stop_reason"], "early_stopping")
            with (folder / "episodes.csv").open(encoding="utf-8") as stream:
                self.assertEqual([int(row["episode"]) for row in csv.DictReader(stream)], list(range(1, 7)))
            with self.assertRaises(ValueError):
                train(replace(config, max_episodes=8, seed=99), folder, resume=True, source_hashes={"test": "smoke"})
            other_algorithm = "double_dqn" if algorithm == "dqn" else "dqn"
            with self.assertRaisesRegex(ValueError, "algorithm"):
                train(replace(config, max_episodes=8, algorithm=other_algorithm), folder,
                      resume=True, source_hashes={"test": "smoke"})

            package = export_policy(folder)
            destination = Path(directory) / "model"
            # This ZIP was just created by the exporter under test, not an external archive.
            with zipfile.ZipFile(package) as archive:
                archive.extractall(destination)
            policy_folder = destination / f"{algorithm}_rook"
            manifest = json.loads((policy_folder / "model.json").read_text())
            self.assertEqual(manifest["id"], f"custom-{algorithm.replace('_', '-')}-rook")
            policy = Policy(policy_folder / "best.pt")
            observation = Game(ENDGAMES["rook"]["fen"]).observation()
            self.assertIn(policy.select_move(observation)["action_id"], observation["legal_action_ids"])
            cases = [observation]
            black = Game(ENDGAMES["rook"]["fen"])
            black.step(black.legal[0])
            cases.append(black.observation())
            requests = "".join(json.dumps({"protocol": "xiangqi-lab.agent.v1", "request_id": i,
                                "time_budget_ms": 800, "observation": obs}) + "\n" for i, obs in enumerate(cases))
            worker = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "xiangqi_lab/model_worker.py"),
                                     str(policy_folder)], input=requests, capture_output=True,
                                    text=True, encoding="utf-8", timeout=60)
            self.assertEqual(worker.returncode, 0, worker.stderr)
            replies = [json.loads(line) for line in worker.stdout.splitlines()]
            self.assertEqual(len(replies), 2)
            for obs, reply in zip(cases, replies):
                self.assertNotIn("error", reply)
                self.assertIn(reply["action_id"], obs["legal_action_ids"])

    def test_whole_notebook_keeps_formal_training_off(self):
        runner = r'''
import json
from pathlib import Path
import sys
nb=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
scope={"__name__":"__main__"}
for cell in nb["cells"]:
    if cell["cell_type"]=="code":
        exec(compile("".join(cell["source"]),cell["metadata"]["id"],"exec"),scope)
assert scope["START_TRAINING"] is False
assert scope["training_result"] is None
assert scope["train_config"].min_episodes >= 1000
assert not (scope["OUTPUT_DIR"] / "runs").exists()
'''
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run([sys.executable, "-X", "utf8", "-c", runner,
                                  str(ROOT / "notebooks/DQN_ChineseChess.ipynb")],
                                 cwd=directory, capture_output=True, text=True, encoding="utf-8", timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
