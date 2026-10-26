"""Evaluate saved best policies after both runs, without training or selection."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import torch
from dqn_model import QNetwork, encode_state, greedy_action
from dqn_training import FixedOpponentEnv


def run_frozen_comparison(base_dir, runs, test_data, reference):
    base_dir = Path(base_dir)
    assert (base_dir / "completed.json").is_file(), "Finish both training runs first."
    folder = base_dir / "test200"
    folder.mkdir(exist_ok=True)
    (folder / "episodes").mkdir(exist_ok=True)
    cases = [(b["id"], seed) for b in test_data["boards"] for seed in b["opponent_seeds"]]
    assert len(cases) == 200 and len(set(cases)) == 200
    training_bank = json.loads((base_dir / "boards.json").read_text())
    training_starts = {" ".join(b["fen"].split()[:2]) for b in training_bank["training"] + training_bank["validation"]}
    assert not training_starts.intersection(" ".join(b["fen"].split()[:2]) for b in test_data["boards"])
    report = {"status": "running", "training": False, "epsilon": 0, "games_per_model": 200,
              "scope": "Previously inspected ten-board regression bank; no test-based checkpoint selection.",
              "results": {}, "references": reference["results"]}
    torch.set_num_threads(2)
    with torch.inference_mode():
        for algorithm, run_name in runs:
            checkpoint = base_dir / run_name / "best.pt"
            before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert saved["config"]["algorithm"] == algorithm
            assert saved["config"]["max_plies"] == 160 and saved["config"]["learner"] == "red"
            model = QNetwork(**saved["network_config"])
            model.load_state_dict(saved["model"])
            model.eval().requires_grad_(False)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            rows = []
            for board in test_data["boards"]:
                for repeat, seed in enumerate(board["opponent_seeds"], 1):
                    env = FixedOpponentEnv(160, "red")
                    obs, info = env.reset(board["fen"], seed)
                    while info["outcome"] is None:
                        action = greedy_action(model, encode_state(obs), obs["legal_action_ids"], "cpu")
                        obs, _, _, _, info = env.step(action)
                    outcome = info["outcome"]
                    result = ("truncations" if outcome["truncated"] else "draws" if outcome["winner"] is None
                              else "wins" if outcome["winner"] == "red" else "losses")
                    rows.append(dict(model=run_name, board=board["id"], group=board["group"], repeat=repeat,
                                     seed=seed, result=result, reason=outcome["reason"],
                                     plies=len(env.game.history), final_fen=env.game.fen))
                    episode = env.game.export()
                    episode["evaluation"] = dict(model=run_name, checkpoint_sha256=before_hash,
                                                 board=board["id"], seed=seed, training=False, epsilon=0)
                    (folder / "episodes" / f"{run_name}_{board['id']}_{repeat:02d}.json").write_text(json.dumps(episode))
                print(f"TEST {run_name}: {len(rows)}/200 games", flush=True)
            assert [(r["board"], r["seed"]) for r in rows] == cases
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before_hash
            with (folder / f"{run_name}_games.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            def stats(selected):
                counts = Counter(r["result"] for r in selected)
                return {**{k: counts[k] for k in ("wins", "losses", "draws", "truncations")},
                        "games": len(selected), "win_rate": counts["wins"] / len(selected),
                        "reasons": dict(Counter(r["reason"] for r in selected))}
            result = {**stats(rows), "best_episode": saved["episode"], "weights_unchanged": True,
                      "checkpoint_sha256": before_hash,
                      "per_group": {g: stats([r for r in rows if r["group"] == g]) for g in ("rook", "cannon", "horse")}}
            previous = reference["results"]["new_" + algorithm]
            result["win_rate_change_pp_vs_uniform20"] = 100 * (result["win_rate"] - previous["win_rate"])
            report["results"][algorithm] = result
            (folder / "comparison.json").write_text(json.dumps(report, indent=2))
            print("TEST RESULT", algorithm, json.dumps(result), flush=True)
    report["status"] = "complete"
    (folder / "comparison.json").write_text(json.dumps(report, indent=2))
    (folder / "boards.json").write_text(json.dumps(test_data, indent=2))
    package = base_dir / (base_dir.name + "_RESULTS.zip")
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(base_dir))
        for _, run_name in runs:
            for path in sorted((base_dir / run_name).glob("*")):
                if path.suffix in (".json", ".csv", ".zip"):
                    archive.write(path, path.relative_to(base_dir))
        for name in ("completed.json", "boards.json", "source_hashes.json"):
            archive.write(base_dir / name, name)
    return package, report
