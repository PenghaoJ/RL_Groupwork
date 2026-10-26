"""Test the DQN reproduction and preserved Double DQN with 200 fresh seeds."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "xiangqi_lab"
if not SOURCE_DIR.is_dir():
    SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

import torch
from dqn_model import MODEL_SCHEMA, QNetwork, encode_state, greedy_action
from dqn_training import ENDGAMES, FixedOpponentEnv, TrainConfig

GAMES = 200
SPLIT = "heldout-20261022-r2-repro-v1"
DDQN_SHA = "b616b5c8fdb10587a8b9054484348ee0db118b02cf2b4e6e5e54cba9d90d7379"


def load_best(path, algorithm, expected_sha):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {algorithm} checkpoint: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            entries = [n for n in archive.namelist() if n.endswith("/best.pt") or n == "best.pt"]
            if len(entries) != 1:
                raise ValueError("The package must contain exactly one best.pt.")
            raw = archive.read(entries[0])
    else:
        if path.name != "best.pt":
            raise ValueError("Use best.pt, not the final/resume checkpoint.")
        raw = path.read_bytes()
    checkpoint_sha = hashlib.sha256(raw).hexdigest()
    if checkpoint_sha != expected_sha:
        raise ValueError(f"Wrong {algorithm} checkpoint hash. No test was run.")
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if saved.get("schema") != MODEL_SCHEMA or saved.get("kind") != "policy":
        raise ValueError("Unsupported policy checkpoint.")
    config = TrainConfig(**saved["config"])
    config.validate()
    if (config.algorithm != algorithm or config.epsilon_decay_steps != 100000
            or config.preset != "rook" or config.learner != "red" or config.seed != 42
            or saved["validation"]["games"] != 50):
        raise ValueError(f"Wrong {algorithm} configuration. No test was run.")
    for name in ("rules.py", "dqn_model.py"):
        # Match the notebook builder's newline normalization on Windows.
        source = (SOURCE_DIR / name).read_text(encoding="utf-8")
        actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if actual != saved.get("source_hashes", {}).get(name):
            raise ValueError(f"Source differs from the saved run: {name}")
    model = QNetwork(**saved["network_config"])
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    metadata = {
        "path": str(path), "checkpoint_sha256": checkpoint_sha,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "best_episode": saved["episode"], "validation": saved["validation"],
        "config": vars(config), "source_hashes": saved["source_hashes"],
    }
    return model, config, metadata


def wilson_interval(wins, total):
    z = 1.959963984540054
    proportion = wins / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [centre - half, centre + half]


@torch.inference_mode()
def test_model(name, model, config, output):
    rows = []
    stats = {"wins": 0, "losses": 0, "draws": 0, "truncations": 0, "environment_plies": 0}
    with (output / f"{name}_games.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["game", "seed", "result", "reason", "plies"])
        writer.writeheader()
        for index in range(GAMES):
            seed = f"{SPLIT}:opponent:{config.seed}:{index}"
            env = FixedOpponentEnv(config.max_plies, config.learner)
            observation, info = env.reset(ENDGAMES[config.preset]["fen"], seed)
            while info["outcome"] is None:
                action = greedy_action(model, encode_state(observation), observation["legal_action_ids"], "cpu")
                observation, _, _, _, info = env.step(action)
            result = info["outcome"]
            key = ("truncations" if result["truncated"] else "draws" if result["winner"] is None
                   else "wins" if result["winner"] == config.learner else "losses")
            plies = len(env.game.history)
            stats[key] += 1
            stats["environment_plies"] += plies
            row = {"game": index + 1, "seed": seed, "result": key, "reason": result["reason"], "plies": plies}
            rows.append(row)
            writer.writerow(row)
            stream.flush()
            if (index + 1) % 25 == 0:
                print(f"{name}: {index + 1}/{GAMES} | W/L/D/T "
                      f"{stats['wins']}/{stats['losses']}/{stats['draws']}/{stats['truncations']}", flush=True)
    return {**stats, "games": GAMES, "split": SPLIT, "win_rate": stats["wins"] / GAMES,
            "score": (stats["wins"] - stats["losses"]) / GAMES,
            "win_rate_wilson_95": wilson_interval(stats["wins"], GAMES)}, rows


def compare(dqn_path, ddqn_path, dqn_sha, output_path, check_only=False):
    torch.set_num_threads(2)
    candidates = {"dqn": load_best(dqn_path, "dqn", dqn_sha),
                  "double_dqn": load_best(ddqn_path, "double_dqn", DDQN_SHA)}
    configs = [{k: v for k, v in vars(item[1]).items() if k != "algorithm"}
               for item in candidates.values()]
    if configs[0] != configs[1]:
        raise ValueError("Training settings differ beyond the algorithm.")
    if check_only:
        print(json.dumps({k: v[2] for k, v in candidates.items()}, indent=2))
        return
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "games_per_model": GAMES, "split": SPLIT,
              "seed_bank": [f"{SPLIT}:opponent:42:{i}" for i in range(GAMES)],
              "opponent": "uniform random legal moves", "device": "cpu",
              "torch_version": str(torch.__version__), "training": False,
              "epsilon": 0, "results": {},
              "scope": "Same fixed rook position, new opponent seeds; not unseen-board or multi-training-seed generalisation."}
    report_path = output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    game_rows = {}
    for name, (model, config, metadata) in candidates.items():
        print(f"Testing {name}: {GAMES} games, no learning.", flush=True)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        stats, game_rows[name] = test_model(name, model, config, output)
        if not all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()):
            raise RuntimeError("Model weights changed during evaluation.")
        if hashlib.sha256(Path(metadata["path"]).read_bytes()).hexdigest() != metadata["file_sha256"]:
            raise RuntimeError("The checkpoint file changed during evaluation.")
        report["results"][name] = {**metadata, **stats, "weights_unchanged": True}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(stats, indent=2), flush=True)
    report["status"] = "complete"
    report["paired_wins"] = {"both_win": 0, "dqn_only": 0, "double_dqn_only": 0, "neither_win": 0}
    for left, right in zip(game_rows["dqn"], game_rows["double_dqn"]):
        assert left["seed"] == right["seed"]
        pair_key = ("both_win" if left["result"] == right["result"] == "wins" else
                    "dqn_only" if left["result"] == "wins" else
                    "double_dqn_only" if right["result"] == "wins" else "neither_win")
        report["paired_wins"][pair_key] += 1
    report["win_rate_difference_dqn_minus_double_dqn"] = (
        report["results"]["dqn"]["win_rate"] - report["results"]["double_dqn"]["win_rate"])
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {report_path.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dqn", required=True, help="Reproduced DQN best.pt or exported ZIP")
    parser.add_argument("--double-dqn", required=True, help="Preserved Double DQN best.pt or exported ZIP")
    parser.add_argument("--dqn-sha256", required=True, help="Hash of the best.pt chosen before testing")
    parser.add_argument("--output", default="output/dqn/DQN_vs_DoubleDQN_TEST200_20261022")
    parser.add_argument("--check-only", action="store_true", help="Check identities; do not play games")
    args = parser.parse_args()
    compare(args.dqn, args.double_dqn, args.dqn_sha256, args.output, args.check_only)


if __name__ == "__main__":
    main()
