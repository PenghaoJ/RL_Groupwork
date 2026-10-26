"""Verify downloaded checkpoints and reproduce the frozen test games locally."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
import torch
from dqn_model import QNetwork, encode_state, greedy_action
from dqn_training import FixedOpponentEnv

BASE = ROOT / "output/dqn/ROOK20_TERMINAL50_20261025_S42"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    package = BASE / (BASE.name + "_RESULTS.zip")
    target = BASE / "results"
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        for entry in archive.infolist():
            dest = (target / entry.filename).resolve()
            assert dest.is_relative_to(target.resolve())
            if entry.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            data = archive.read(entry)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert dest.read_bytes() == data
            else:
                dest.write_bytes(data)
    completed = read(target / "completed.json")
    comparison = read(target / "test200/comparison.json")
    old = read(ROOT / "output/dqn/UNSEEN10_FOUR_20261025/results/comparison.json")
    assert comparison["references"] == old["results"]
    assert completed["status"] == comparison["status"] == "complete"
    assert comparison["training"] is False and comparison["epsilon"] == 0
    bank = read(target / "test200/boards.json")
    assert bank == read(ROOT / "output/dqn/UNSEEN10_FOUR_20261025/boards.json")
    training_bank = read(target / "boards.json")
    starts = {" ".join(b["fen"].split()[:2]) for b in training_bank["training"] + training_bank["validation"]}
    assert not starts.intersection(" ".join(b["fen"].split()[:2]) for b in bank["boards"])
    preflight = read(BASE / "preflight.json")
    assert completed["board_bank_sha256"] == preflight["board_bank_sha256"]
    assert read(target / "source_hashes.json") == preflight["source_hashes"]
    manifest = read(BASE / "baselines/manifest.json")
    for meta in manifest.values():
        for path in (Path(meta["original_path"]), BASE / "baselines" / Path(meta["original_path"]).name):
            assert sha(path.read_bytes()) == meta["file_sha256"], str(path)
    torch.set_num_threads(2)
    local_models, verified, summaries = [], {}, {}
    for algorithm, label in (("dqn", "DQN"), ("double_dqn", "DoubleDQN")):
        name = f"{label}_{BASE.name}"
        folder = target / name
        summary = read(folder / "summary.json")
        summaries[algorithm] = summary
        assert summary == completed["runs"][name]
        config = read(folder / "config.json")
        expected_config = preflight["configs"][algorithm]
        assert {k: config[k] for k in expected_config} == expected_config
        assert config["source_hashes"] == preflight["source_hashes"]
        assert config["actual_device"] == "cpu" and config["opponent"] == "uniform-legal-random-v1"
        sampling = read(folder / "sampling.json")
        assert sampling["mixture_batches"] == summary["updates"]
        assert sampling["mode"] == "terminal_mixture" and sampling["terminal_mix"] == .5
        policy = next(folder.glob("*_model.zip"))
        with zipfile.ZipFile(policy) as archive:
            raw = archive.read(next(n for n in archive.namelist() if n.endswith("/best.pt")))
        digest = sha(raw)
        assert digest == comparison["results"][algorithm]["checkpoint_sha256"]
        saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        assert saved["episode"] == summary["best_episode"] == 300
        assert saved["config"] == expected_config
        assert saved["board_bank_sha256"] == completed["board_bank_sha256"]
        destination = BASE / "models" / f"{name}_BEST_EP0300.zip"
        if destination.exists():
            assert destination.read_bytes() == policy.read_bytes()
        else:
            destination.write_bytes(policy.read_bytes())
        local_models.append(dict(algorithm=algorithm, path=str(destination.relative_to(BASE)),
                                 checkpoint_sha256=digest, file_sha256=sha(destination.read_bytes())))
        model = QNetwork(**saved["network_config"])
        model.load_state_dict(saved["model"])
        model.eval().requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rows = list(csv.DictReader((target / "test200" / f"{name}_games.csv").open(newline="")))
        expected = [(b["id"], seed) for b in bank["boards"] for seed in b["opponent_seeds"]]
        assert len(expected) == len(rows) == 200 and len(set(expected)) == 200
        assert [(r["board"], r["seed"]) for r in rows] == expected
        lookup = {b["id"]: b for b in bank["boards"]}
        counts, reasons, plies = Counter(), Counter(), 0
        with torch.inference_mode():
            for index, row in enumerate(rows, 1):
                episode_path = target / "test200/episodes" / f"{name}_{row['board']}_{int(row['repeat']):02d}.json"
                episode = read(episode_path)
                assert episode["evaluation"]["checkpoint_sha256"] == digest
                assert episode["evaluation"]["training"] is False
                assert episode["initial_fen"] == lookup[row["board"]]["fen"]
                env = FixedOpponentEnv(160, "red")
                obs, info = env.reset(episode["initial_fen"], row["seed"])
                while info["outcome"] is None:
                    action = greedy_action(model, encode_state(obs), obs["legal_action_ids"], "cpu")
                    obs, _, _, _, info = env.step(action)
                assert [m["move"] for m in env.game.history] == [m["move"] for m in episode["moves"]], (algorithm, row["board"], row["repeat"])
                assert env.game.fen == row["final_fen"] and len(env.game.history) == int(row["plies"])
                outcome = env.game.outcome()
                assert outcome == episode["outcome"]
                result = "truncations" if outcome["truncated"] else "draws" if outcome["winner"] is None else "wins" if outcome["winner"] == "red" else "losses"
                assert result == row["result"] and outcome["reason"] == row["reason"]
                counts[result] += 1
                reasons[row["reason"]] += 1
                plies += len(env.game.history)
                if index % 50 == 0:
                    print(f"Verified {algorithm}: {index}/200 exact local reproductions", flush=True)
        for key in ("wins", "losses", "draws", "truncations"):
            assert counts[key] == comparison["results"][algorithm][key]
        assert dict(reasons) == comparison["results"][algorithm]["reasons"]
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        verified[algorithm] = dict(exact_reproductions=200, legal_plies=plies, weights_unchanged=True, checkpoint_sha256=digest)
    audit = dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
                 downloaded_bundle_sha256=sha(package.read_bytes()), models=local_models,
                 runs=summaries, reproduced_games=400, reproductions=verified,
                 four_originals_unchanged=True, cases_identical_to_previous200=True,
                 training_validation_test_starts_disjoint=True, configs_verified=True,
                 cloud_torch="2.11.0+cpu", local_torch=torch.__version__, training=False,
                 note="completed.json was written before testing; test200/comparison.json is the final test status.")
    write(BASE / "results_verification.json", audit)
    write(BASE / "latest-backups.json", local_models)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
