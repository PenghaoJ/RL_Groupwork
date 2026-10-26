"""Bundle current source, frozen opponents, board splits and MCTS handoff notes."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = "Xiangqi_MCTS_Handoff_20261026"
OUT = ROOT / "output/handoff"
REQUIREMENTS = "torch>=2.8,<3\nnumpy==2.2.6\ngymnasium==1.2.1\nstable-baselines3==2.7.0\nsb3-contrib==2.7.0\n"
MODELS = {
    "local_match_double_dqn": {
        "files": ["best.pt", "dqn_model.py", "policy.py", "policy_perspective.py", "model.json"],
        "checkpoint": "best.pt", "algorithm": "Double DQN", "best_episode": 200,
        "sha256": "b616b5c8fdb10587a8b9054484348ee0db118b02cf2b4e6e5e54cba9d90d7379",
    },
    "local_ppo_rook20_seed42": {
        "files": ["weights.zip", "policy.py", "policy_perspective.py", "model.json", "encoder.json", "ppo_encoding.py", "ppo_features.py"],
        "checkpoint": "weights.zip", "algorithm": "PPO", "seed": 42, "best_episode": 700,
        "sha256": "0f06447b8c97fd433b425a65f2427f055ccda65ab79fe173fd7f3e2a98ee3876",
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def starter_notebook():
    cells = []

    def cell(kind, text):
        item = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", "# MCTS environment starter\nExtract the handoff ZIP first. This notebook checks the shared rules; it does not implement MCTS or train a network.\n")
    cell("markdown", "## 1. Select the extracted project\nRun beside the extracted project, or set PROJECT_ROOT to its path in your cloud runtime. No connection to the owner's website is needed.\n")
    cell("code", '''from pathlib import Path
import sys

candidates = [Path.cwd(), Path.cwd().parent, Path("/content/Xiangqi_MCTS_Handoff_20261026")]
PROJECT_ROOT = next((p for p in candidates if (p / "xiangqi_lab/rules.py").is_file()), None)
assert PROJECT_ROOT is not None, "Set PROJECT_ROOT to the extracted handoff folder."
sys.path.insert(0, str(PROJECT_ROOT / "xiangqi_lab"))
print(PROJECT_ROOT)
''')
    cell("markdown", "## 2. Load training and validation starts\nKeep test boards out of this step.\n")
    cell("code", '''import json
from board_bank import BoardBank
from training import XiangqiEnv

data = json.loads((PROJECT_ROOT / "experiments/rook20_20261025/boards.json").read_text(encoding="utf-8"))
bank = BoardBank(data)
print("Training:", len(bank.training), "Validation:", len(bank.validation))
env = XiangqiEnv(max_plies=160)
start = bank.training_position(episode=1, seed=42)
observation, info = env.reset(fen=start["fen"], seed=42)
print(start["id"], observation["side_to_move"], len(observation["legal_moves"]))
''')
    cell("markdown", "## 3. Simulate on a copy\nDeep copying preserves the current game's repetition counts and elapsed moves.\n")
    cell("code", '''import copy

before = env.game.export()
simulation = copy.deepcopy(env.game)
move = simulation.legal[0]
outcome = simulation.step(move, actor="simulation-example")
assert env.game.export() == before
print("Simulated:", move, "Outcome:", outcome)
print("Real board unchanged:", env.game.fen)
''')
    cell("markdown", "## 4. Implement your search\nUse docs/MCTS_HANDOFF.md and MODEL_API.md. Add selection, expansion, evaluation and value backup to a new policy. Test history-sensitive state reconstruction before using the website adapter. The first-legal-move template is not MCTS.\n")
    return dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, nbformat=4, nbformat_minor=5)


START = r'''# MCTS teammate package — start here

This is the current source and two frozen opponents, prepared on 26 October 2026 (Australia/Sydney). It includes local work not yet pushed to GitHub. Keep this folder separate from an existing checkout until you have compared changes.

1. Give your AI `docs/MCTS_AI_PROMPT.txt` and access to this extracted folder.
2. Read `docs/EXPERIMENT_STATUS_20261026.md` for what was done and measured.
3. Read `docs/MCTS_HANDOFF.md` for your search/learning task and exact data boundaries.
4. Start with `notebooks/MCTS_Handoff_Start.ipynb` or the commands below. A working MCTS trainer is not included; your task is to implement it.

## Local setup

Install Python 3.12 if needed. From this extracted folder on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe tools/check_mcts_handoff.py
```

On macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python tools/check_mcts_handoff.py
chmod +x xiangqi_lab/engine/Pikafish-MacOS-universal
```

The first check needs only the standard library. It verifies hashes, board splits, state cloning and the template worker. It performs no training.

To use the frozen neural opponents, install their dependencies, then check and launch the website:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-opponents.txt
.\.venv\Scripts\python.exe tools/check_mcts_handoff.py --with-opponents
.\.venv\Scripts\python.exe -X utf8 xiangqi_lab/app.py --port 8766
```

On macOS replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Use CPU first. Requirements allow Torch 2.8 or newer below 3; the exact tested Windows runtime was Python 3.12.14 / Torch 2.14.1+cpu. The PPO source package specified Torch 2.8.0, and the imported actor was verified locally using the newer runtime. This is not a locked, tested environment for every OS. Save your resolved package versions and rerun preflight on your machine. Existing cloud Torch need not be replaced if it is compatible.

Select `Double DQN (our model)` and `PPO (seed 42)` to watch them play. Add your own uniquely named model folder when ready. Neural dependencies are not included in a portable runtime; the root BAT launchers belong to older app releases and are not the setup path for this handoff.

## Cloud use

Upload this ZIP to your own storage and extract it in the chosen runtime, then point the starter notebook at the extracted folder. Rules, board splits and simulations run without the browser or Pikafish. Linux cloud has no bundled Pikafish executable. Use a runtime permitted for the work; this handoff starts no paid resources or long training jobs. Preserve logs and checkpoints outside temporary runtime storage.

## Included

- Current application source, rules, tests, DQN reference notebooks and MCTS starter notebook.
- Twenty training / six validation rook starts; ten evaluation-only starts with the existing opponent seed bank.
- Old selected Double DQN and PPO seed 42 inference weights and adapters. Only copied manifests use portable `{python}`; original local files were not changed.
- Experiment evidence, 20 direct-match replays, and one separate website demonstration.
- Windows/macOS Pikafish binaries, NNUE, source archive and original notices. Preserve component licenses when redistributing. Team original code has no separate public license; this is a team handoff.

There is no MCTS checkpoint, full PPO trainer, training-resume state, Python installation, Git metadata, private virtual environment or old Windows-only legacy GUI in this ZIP. Other historical model weights remain with the owner.

Do not train or tune on `output/dqn/UNSEEN10_FOUR_20261025/boards.json`. It is only for frozen evaluation. Train/validation starts are in `experiments/rook20_20261025/boards.json`.

The copied source uses the same model IDs and paths as the match runner. After dependency setup, you can reproduce the existing paired match with:

```text
python tools/match_double_dqn_ppo.py --output output/matches/my_reproduction
```

Expected saved result: Double DQN 0 wins, PPO 2 wins, 18 repetition draws, 0 limits. The output folder must be new. See `output/matches/DoubleDQN_vs_PPO42_20261026/report.html`. Reproduction with a different dependency stack is a check, not an assumption.
'''


def main():
    target = OUT / NAME
    archive = OUT / (NAME + ".zip")
    if target.exists() or archive.exists():
        raise FileExistsError("Preserve existing handoffs; use a new package version.")
    write_json(ROOT / "notebooks/MCTS_Handoff_Start.ipynb", starter_notebook())
    target.mkdir(parents=True)
    paths = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT).decode("utf-8").split("\0")
    prefixes = {"xiangqi_lab", "tools", "docs", "experiments", "notebooks", ".github"}
    root_files = {"README.md", "CONTRIBUTING.md", ".gitignore", ".gitattributes"}
    source_hashes = {}
    for name in sorted(set(filter(None, paths))):
        path = Path(name)
        if path.parts[0] not in prefixes and name not in root_files:
            continue
        if any(p in {"__pycache__", ".venv", ".setup", ".git", "runtime"} or p.startswith("local_") for p in path.parts):
            continue
        src, dst = ROOT / path, target / path
        if not src.is_file() or src.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        source_hashes[path.as_posix()] = sha(src)
    for name, config in MODELS.items():
        source = ROOT / "xiangqi_lab/models" / name
        destination = target / "xiangqi_lab/models" / name
        destination.mkdir(parents=True)
        assert sha(source / config["checkpoint"]) == config["sha256"]
        for file in config["files"]:
            shutil.copy2(source / file, destination / file)
        manifest = json.loads((destination / "model.json").read_text(encoding="utf-8"))
        manifest["python"] = "{python}"
        write_json(destination / "model.json", manifest)
        write_json(destination / "handoff-provenance.json", {k: v for k, v in config.items() if k != "files"})
        (destination / "README.md").write_text("Frozen opponent, not a resumable trainer. See ../../../START_HERE.md and ../../../docs/MCTS_HANDOFF.md. Weights are unchanged; the copied model.json uses the server interpreter. Trained as Red; Black uses a tested view transform.\n", encoding="utf-8")
        (destination / "requirements.txt").write_text(REQUIREMENTS if "ppo" in name else "torch>=2.8,<3\n", encoding="utf-8")
    copies = {
        "output/dqn/UNSEEN10_FOUR_20261025/boards.json": "output/dqn/UNSEEN10_FOUR_20261025/boards.json",
        "output/dqn/UNSEEN10_FOUR_20261025/results/comparison.json": "handoff/evidence/four_models_comparison.json",
        "output/dqn/ROOK20_TERMINAL50_20261025_S42/results/test200/comparison.json": "handoff/evidence/terminal50_comparison.json",
        "output/dqn/ROOK20_TERMINAL50_20261025_S42/results_verification.json": "handoff/evidence/terminal50_verification.json",
        "output/dqn/DIAGNOSIS_ROOK20_20261025/diagnosis.json": "handoff/evidence/reward_repetition_diagnosis.json",
        "output/ppo/RECEIVED_20261026/verification.json": "handoff/evidence/ppo_verification.json",
    }
    for source, destination in copies.items():
        dst = target / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, dst)
    match_dir = Path("output/matches/DoubleDQN_vs_PPO42_20261026")
    shutil.copytree(ROOT / match_dir, target / match_dir)
    (target / "START_HERE.md").write_text(START, encoding="utf-8")
    (target / "requirements-opponents.txt").write_text(REQUIREMENTS, encoding="utf-8")
    write_json(target / "SNAPSHOT.json", {
        "package": NAME, "local_date": "2026-10-26", "timezone": "Australia/Sydney",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "includes_uncommitted_work": True, "pushed_by_this_task": False,
        "source_sha256": source_hashes, "opponents": MODELS,
        "omitted": ["Git history", "local runtimes", "other historical weights", "PPO trainer", "MCTS implementation", "optimizer/replay/RNG resume state"],
    })
    files = sorted(p for p in target.rglob("*") if p.is_file())
    (target / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(target).as_posix()}\n" for p in files), encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(p for p in target.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo.from_file(path, arcname=NAME + "/" + path.relative_to(target).as_posix())
            info.create_system = 3
            mode = 0o755 if path.name == "Pikafish-MacOS-universal" or path.suffix == ".command" else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            output.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(archive) as check:
        assert check.testzip() is None
    digest = sha(archive)
    archive.with_suffix(".zip.sha256.txt").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest, "files": len(files) + 1}, indent=2))


if __name__ == "__main__":
    main()
