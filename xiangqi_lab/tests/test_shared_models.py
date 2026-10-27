"""Published agents must use shared files, preserve weights and return legal moves."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from rules import Game, PRESETS
from model_registry import models, command_for
from engines import check_model, choose_move, close_workers
import start


def load_mcts():
    folder = APP / "models/mcts_v4"
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location("published_mcts", folder / "policy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Policy()
    finally:
        sys.path.remove(str(folder))


class SharedModelTests(unittest.TestCase):
    def tearDown(self):
        close_workers()

    def test_payload_hashes_and_portable_manifests(self):
        provenance = json.loads((APP / "models/shared_provenance.json").read_text())
        for name, expected in provenance["sha256"].items():
            self.assertEqual(hashlib.sha256((APP / "models" / name).read_bytes()).hexdigest(), expected)
        entries = {m["id"]: m for m in models()}
        for mid in ("custom-double-dqn", "custom-ppo", "custom-mcts-v4"):
            entry = entries[mid]
            self.assertEqual(entry["python"], "{python}")
            self.assertEqual(command_for(entry)[0], sys.executable)

    def test_mcts_scene_follows_each_game_start_and_black_first_request(self):
        policy = load_mcts()
        for scene in (0, 2, 1, 3, 0):
            game = Game(PRESETS[scene]["fen"])
            game.step(game.legal[0])
            self.assertEqual(game.side, "black")
            policy._identify_scene(game.observation())
            policy._load_experience()
            self.assertEqual(policy.scene_id, scene)
            self.assertTrue(policy.experience)
        game = Game("4k4/9/9/9/2R6/9/9/9/9/3K5 w - - 0 1")
        policy._identify_scene(game.observation())
        policy._load_experience()
        self.assertIsNone(policy.scene_id)
        self.assertEqual(policy.experience, {})
        legacy = Game(PRESETS[0]["fen"])
        legacy.step(legacy.legal[0])
        observation = legacy.observation()
        del observation["initial_fen"]
        policy._identify_scene(observation)
        self.assertIsNone(policy.scene_id)

    def test_mcts_worker_moves_on_both_sides(self):
        self.assertTrue(check_model("custom-mcts-v4")["passed"])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("sb3_contrib"), "Optional CPU model dependencies not installed")
    def test_neural_workers_on_all_presets_and_both_sides(self):
        for mid in ("custom-double-dqn", "custom-ppo"):
            for preset in PRESETS[:4]:
                game = Game(preset["fen"])
                for _ in range(2):
                    move, _ = choose_move(mid, game, 100)
                    self.assertIn(move, game.legal)
                    game.step(move)


class ModelStartupTests(unittest.TestCase):
    def test_missing_dependencies_check_does_not_install(self):
        with patch.object(start, "ready", return_value=False), patch.object(start.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "dependencies are missing"):
                start.prepare(allow_install=False)
            run.assert_not_called()

    def test_install_failure_does_not_open_server(self):
        with patch.object(sys, "argv", ["start.py"]), patch.object(start, "prepare", side_effect=RuntimeError("offline")), patch.object(start.subprocess, "call") as call:
            self.assertEqual(start.main(), 1)
            call.assert_not_called()

    def test_ready_environment_is_reused_without_installing(self):
        with patch.object(start, "ready", return_value=True), patch.object(start.subprocess, "run") as run:
            self.assertEqual(start.prepare(), start.runtime_python(start.ENV))
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
