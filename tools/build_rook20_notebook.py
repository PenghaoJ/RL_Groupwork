"""Build a self-contained Colab experiment with a frozen board bank."""
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("rules.py", "training.py", "notebook_tools.py", "dqn_model.py", "dqn_training.py", "board_bank.py")


def build(terminal_mixture=False):
    sources = {name: (ROOT / "xiangqi_lab" / name).read_text(encoding="utf-8") for name in NAMES}
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}
    data = json.loads((ROOT / "experiments/rook20_20261025/boards.json").read_text())
    cells = []

    def add(kind, name, text):
        cell = {"cell_type": kind, "metadata": {"id": name}, "source": textwrap.dedent(text).strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            cell["metadata"]["cellView"] = "form"
        cells.append(cell)

    add("markdown", "title", """
        # Xiangqi — 20 rook training boards

        Train DQN and Double DQN from the same seed. Use 20 training boards and 6 separate validation boards. Keep the old models and ten test boards separate.
    """)
    add("markdown", "setup-note", """
        ## 1. Load rules and boards

        Loads the project source snapshot and checks the fixed board split.
    """)
    add("code", "setup", f'''# @title Load rules and boards
import hashlib, importlib, json, sys
from pathlib import Path
SOURCE_FILES = {sources!r}
SOURCE_HASHES = {hashes!r}
BOARD_DATA = {data!r}
EXPERIMENT = "ROOK20_20261025_S42"
SOURCE_DIR = Path("/content/xiangqi_rook20_source")
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
for name, source in SOURCE_FILES.items():
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_HASHES[name]
    path = SOURCE_DIR / name
    if path.exists() and path.read_text() != source:
        raise RuntimeError("Source changed. Start a fresh runtime.")
    path.write_text(source, encoding="utf-8")
    loaded = sys.modules.get(path.stem)
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != path.resolve():
        raise RuntimeError("Another project snapshot is loaded. Use a fresh runtime.")
sys.path.insert(0, str(SOURCE_DIR))
importlib.invalidate_caches()
from board_bank import BoardBank
from rules import Game, make_fen
from notebook_tools import board_svg
from dqn_training import TrainConfig, train, export_policy
import torch
from IPython.display import HTML, SVG, display
bank = BoardBank(BOARD_DATA)
assert len(bank.training) == 20 and len(bank.validation) == 6
for item in bank.training + bank.validation:
    probe = Game(item["generation"]["base_fen"])
    for move in item["generation"]["moves"]:
        probe.step(move)
    assert make_fen(probe.board, "red") == item["fen"]
BASE_DIR = Path("/content/xiangqi_output") / EXPERIMENT
BASE_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "boards.json").write_text(json.dumps(BOARD_DATA, indent=2))
(BASE_DIR / "source_hashes.json").write_text(json.dumps(SOURCE_HASHES, indent=2))
print("20 training boards; 6 validation boards; old ten test starts excluded.")
print("Board bank SHA256:", bank.sha256)
print("PyTorch:", torch.__version__, "| Device: CPU")
print("No training has started.")
''')
    add("markdown", "options-note", """
        ## 2. Set the run

        Sets the episode limit, validation schedule and early stopping. Leave training off when only checking the notebook.
    """)
    add("code", "options", '''# @title Training settings
START_TRAINING = False  # @param {type:"boolean"}
RESUME = False  # @param {type:"boolean"}
RUNS = [("dqn", "DQN_ROOK20_20261025_S42"),
        ("double_dqn", "DoubleDQN_ROOK20_20261025_S42")]
CONFIGS = {algorithm: TrainConfig(
    algorithm=algorithm, preset="rook", learner="red", min_episodes=1000,
    max_episodes=3000, max_plies=160, eval_every=100, eval_games=60,
    patience=5, min_delta=0.02, seed=42, epsilon_decay_steps=100000, device="cpu"
) for algorithm, _ in RUNS}
for config in CONFIGS.values():
    config.validate()
print("Each model: 1000 to 3000 episodes; check every 100 episodes.")
print("Each check: 6 validation boards x 10 seeds = 60 games.")
print("Stop after 5 checks without a gain above 0.02, after at least 1000 episodes.")
print("Balanced training-board order and the same opponent seeds for both models.")
print("Fresh seed-42 initialization; no old checkpoint is loaded.")
''')
    add("markdown", "train-note", """
        ## 3. Train both models

        Runs DQN, then Double DQN. Saves each new best model separately and provides backup links. No test games are run.
    """)
    add("code", "train-both", '''# @title Train both models and keep backups
import base64, csv, io, zipfile
from contextlib import redirect_stdout

class Tee:
    def __init__(self, output, file):
        self.output, self.file = output, file
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        return self.output.write(text)
    def flush(self):
        self.output.flush()
        self.file.flush()

def launch_rook20():
    if not START_TRAINING:
        print("Training is off. Check START_TRAINING to begin.")
        return
    backups = {}
    view = display(HTML("<p>Waiting for the first best checkpoint.</p>"), display_id=True)
    def publish(run_dir, run_name, algorithm):
        policy_id = "custom-" + run_name.lower().replace("_", "-")
        package = export_policy(run_dir, model_id=policy_id, model_name=run_name)
        saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        label = f"{run_name}_BEST_EP{saved['episode']:04d}.zip"
        path = BASE_DIR / label
        path.write_bytes(package.read_bytes())
        meta = {"run": run_name, "algorithm": algorithm, "best_episode": saved["episode"],
                "checkpoint_sha256": hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest(),
                "zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "validation": saved["validation"], "board_bank_sha256": bank.sha256}
        (BASE_DIR / f"{run_name}_backup.json").write_text(json.dumps(meta, indent=2))
        backups[run_name] = (path, meta)
        blocks = ["<h3>Saved best models</h3><p>New experiment only. Old model files are unchanged.</p>"]
        for name, (file, info) in backups.items():
            encoded = base64.b64encode(file.read_bytes()).decode("ascii")
            blocks.append(f'<p><b>{name}</b> | best episode {info["best_episode"]} | validation {info["validation"]["wins"]}/60 wins<br>'
                          f'<a download="{file.name}" data-sha256="{info["zip_sha256"]}" href="data:application/zip;base64,{encoded}">Save {name}</a><br>'
                          f'<small>Checkpoint SHA256: {info["checkpoint_sha256"]}</small></p>')
        view.update(HTML("".join(blocks)))
        print("BACKUP", label, "checkpoint_sha256", meta["checkpoint_sha256"], flush=True)

    for algorithm, run_name in RUNS:
        folder = BASE_DIR / run_name
        done = folder / "summary.json"
        if done.exists():
            summary = json.loads(done.read_text())
            if summary["stop_reason"] in ("early_stopping", "max_episodes"):
                saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
                assert saved["config"] == vars(CONFIGS[algorithm])
                assert saved["board_bank_sha256"] == bank.sha256
                assert saved["source_hashes"] == SOURCE_HASHES
                publish(folder, run_name, algorithm)
                print("Already complete:", run_name, "| No training repeated.", flush=True)
                continue
        resume = bool(RESUME and (folder / "last.pt").exists())
        def on_checkpoint(path, improved):
            if improved:
                publish(path, run_name, algorithm)
        with (BASE_DIR / f"{run_name}.log").open("a", encoding="utf-8") as log:
            with redirect_stdout(Tee(sys.stdout, log)):
                print("START", run_name, "| Resume:", resume, flush=True)
                result = train(CONFIGS[algorithm], folder, resume=resume, source_hashes=SOURCE_HASHES,
                               board_bank=bank, on_checkpoint=on_checkpoint)
                if result["summary"]["stop_reason"] == "interrupted":
                    print("Stopped. Enable RESUME to continue this run.", flush=True)
                    return
                publish(folder, run_name, algorithm)
    summaries = {name: json.loads((BASE_DIR / name / "summary.json").read_text()) for _, name in RUNS}
    (BASE_DIR / "completed.json").write_text(json.dumps({"status": "complete", "runs": summaries,
        "board_bank_sha256": bank.sha256, "test_games_run": 0}, indent=2))
    print("BOTH RUNS COMPLETE. Best models are saved. No test boards were used.", flush=True)
    for name, info in summaries.items():
        print(name, "| trained", info["episodes_completed"], "| best episode", info["best_episode"],
              "| best validation score", info["best_validation_score"], flush=True)

launch_rook20()
''')
    notebook = {"cells": cells, "metadata": {"colab": {"name": "Xiangqi_Rook20_DQN_DoubleDQN_20261025.ipynb"},
                "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
                "nbformat": 4, "nbformat_minor": 0}
    if terminal_mixture:
        # Keep the baseline notebook intact; change only replay sampling.
        for cell in cells:
            text = "".join(cell["source"])
            if cell["metadata"]["id"] == "setup":
                text = text.replace('EXPERIMENT = "ROOK20_20261025_S42"',
                                    'EXPERIMENT = "ROOK20_TERMINAL50_20261025_S42"')
                text = text.replace('SOURCE_DIR = Path("/content/xiangqi_rook20_source")',
                                    'SOURCE_DIR = Path("/content/xiangqi_terminal50_source")')
            if cell["metadata"]["id"] == "title":
                text = "# Xiangqi — terminal replay comparison\n\nSame 20 training boards and 6 validation boards. Only replay sampling changes."
            if cell["metadata"]["id"] == "options":
                text = text.replace("ROOK20_20261025_S42", "ROOK20_TERMINAL50_20261025_S42")
                text = text.replace('epsilon_decay_steps=100000, device="cpu"',
                    'epsilon_decay_steps=100000, device="cpu",\n    replay_sampling="terminal_mixture", terminal_mix=0.5')
                text += '\nprint("Replay: 50% uniform; 50% terminal win/loss/draw groups; exact importance weights.")\n'
            cell["source"] = text.splitlines(keepends=True)
        notebook["metadata"]["colab"]["name"] = "Xiangqi_Rook20_Terminal50_DQN_DoubleDQN_20261025.ipynb"
    path = ROOT / "notebooks" / notebook["metadata"]["colab"]["name"]
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)
    return path


if __name__ == "__main__":
    build()
