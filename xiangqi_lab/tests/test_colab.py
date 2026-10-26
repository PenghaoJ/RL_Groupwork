"""Execute the shipped notebook and exchange real episodes with the website API."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
from app import Handler, LocalHTTPServer
from notebook_tools import ENDGAMES, board_svg, replay_episode
from rules import Game


RUN_NOTEBOOK = r'''
import json
from pathlib import Path
import sys
notebook = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
scope = {"__name__": "__main__"}
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    if "dqn" in cell["metadata"].get("tags", []):
        continue  # The stdlib suite covers the environment; test_dqn covers training.
    source = "".join(cell["source"])
    if cell["metadata"]["id"] == "choose-board":
        source = source.replace('PRESET_ID = "rook"', 'PRESET_ID = ' + repr(sys.argv[2]), 1)
    if cell["metadata"]["id"] == "import":
        source = source.replace('IMPORT_PATH = ""', 'IMPORT_PATH = ' + repr(sys.argv[3]), 1)
    exec(compile(source, cell["metadata"]["id"], "exec"), scope)
assert scope["imported_game"].fen == scope["imported_episode"]["final_fen"]
assert not any(name in sys.modules for name in ("app", "engines", "torch", "gymnasium"))
Path("notebook-result.json").write_text(json.dumps(scope["episode"]), encoding="utf-8")
'''


class ColabTests(unittest.TestCase):
    def test_snapshot_matches_sources_and_has_no_saved_outputs(self):
        notebook = json.loads((ROOT / "notebooks/DQN_ChineseChess.ipynb").read_text(encoding="utf-8"))
        for name, expected in notebook["metadata"]["xiangqi_lab"]["source_sha256"].items():
            source = (ROOT / "xiangqi_lab" / name).read_text(encoding="utf-8")
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected,
                             "Rebuild the Colab notebook after changing " + name)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_run_all_and_http_exchange_for_each_preset(self):
        server = LocalHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()

        def call(endpoint, payload):
            request = urllib.request.Request(
                f"http://127.0.0.1:{server.server_port}/api/{endpoint}",
                json.dumps(payload).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        try:
            for preset_id, preset in ENDGAMES.items():
                with self.subTest(preset=preset_id), tempfile.TemporaryDirectory() as directory:
                    # Produce the input through the actual website endpoints.
                    session = call("session", {})
                    payload = {"session": session["session"], "revision": 0}
                    state = call("reset", {**payload, "fen": preset["fen"]})["state"]
                    payload["revision"] = state["revision"]
                    state = call("move", {**payload, "move": state["legal_moves"][0]})["state"]
                    payload["revision"] = state["revision"]
                    website_episode = call("export", payload)
                    import_path = Path(directory) / "website-game.json"
                    import_path.write_text(json.dumps(website_episode), encoding="utf-8")

                    # Fresh process, no project sys.path: the notebook must stand alone.
                    run = subprocess.run(
                        [sys.executable, "-X", "utf8", "-c", RUN_NOTEBOOK,
                         str(ROOT / "notebooks/DQN_ChineseChess.ipynb"), preset_id, str(import_path)],
                        cwd=directory, capture_output=True, text=True, encoding="utf-8", timeout=60)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    episode = json.loads((Path(directory) / "notebook-result.json").read_text(encoding="utf-8"))
                    self.assertEqual(episode["initial_fen"], Game(preset["fen"]).fen)
                    self.assertEqual(len(episode["moves"]), 7)

                    # Import cloud output into the real API, then export it again.
                    imported = call("import", {**payload, "episode": episode})["state"]
                    payload["revision"] = imported["revision"]
                    exported = call("export", payload)
                    for field in ("initial_fen", "final_fen", "moves", "max_plies", "outcome"):
                        self.assertEqual(exported[field], episode[field], field)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)

    def test_replay_rejects_corrupt_game_and_board_svg_is_valid(self):
        game = Game(ENDGAMES["rook"]["fen"])
        game.step(game.legal[0])
        episode = game.export()
        with self.assertRaises(ValueError):
            replay_episode({**episode, "final_fen": "bad"})
        with self.assertRaises(ValueError):
            replay_episode({**episode, "moves": [{"move": "a0a0"}]})
        svg = board_svg(game, "Rook & pawn <demo>")
        ET.fromstring(svg)
        self.assertIn("Rook &amp; pawn &lt;demo&gt;", svg)
        self.assertIn("帥", svg)
        self.assertIn("將", svg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
