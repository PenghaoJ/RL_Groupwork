"""Exercise the automatic result bundle with a short no-learning test limit."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
sys.path.insert(0, str(ROOT / "tools"))
import torch
from dqn_model import QNetwork
from dqn_training import FixedOpponentEnv
import terminal_replay_evaluation as evaluation


class TerminalEvaluationTests(unittest.TestCase):
    def test_bundle_counts_and_frozen_weights(self):
        test_root = ROOT / "output/dqn/UNSEEN10_FOUR_20261025"
        if not test_root.exists():
            self.skipTest("The frozen local evaluation bank is not in the repository.")
        bank = json.loads((test_root / "boards.json").read_text())
        reference = json.loads((test_root / "results/comparison.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / "completed.json").write_text('{"status":"complete"}')
            (base / "boards.json").write_bytes((ROOT / "experiments/rook20_20261025/boards.json").read_bytes())
            (base / "source_hashes.json").write_text('{}')
            runs = [("dqn", "test_dqn"), ("double_dqn", "test_double_dqn")]
            hashes = {}
            torch.manual_seed(12)
            model = QNetwork()
            for algorithm, name in runs:
                folder = base / name
                folder.mkdir()
                saved = dict(config={"algorithm": algorithm, "max_plies": 160, "learner": "red"},
                             episode=100, model=model.state_dict(), network_config=model.config)
                torch.save(saved, folder / "best.pt")
                hashes[name] = hashlib.sha256((folder / "best.pt").read_bytes()).hexdigest()
            # Use a short limit only in this pipeline test; production remains 160.
            with patch.object(evaluation, "FixedOpponentEnv", side_effect=lambda *_: FixedOpponentEnv(4, "red")):
                package, report = evaluation.run_frozen_comparison(base, runs, bank, reference)
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["training"])
            for algorithm, name in runs:
                row = report["results"][algorithm]
                self.assertEqual(row["games"], 200)
                self.assertEqual(sum(row[k] for k in ("wins", "losses", "draws", "truncations")), 200)
                self.assertEqual(row["checkpoint_sha256"], hashes[name])
                self.assertTrue(row["weights_unchanged"])
                self.assertEqual([row["per_group"][g]["games"] for g in ("rook", "cannon", "horse")], [80, 60, 60])
            with zipfile.ZipFile(package) as archive:
                self.assertEqual(sum('/episodes/' in n for n in archive.namelist()), 400)
                self.assertIn("test200/comparison.json", archive.namelist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
