"""Reproduce the delivered PPO games and check local model adapters."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
import torch
from dqn_training import FixedOpponentEnv
from policy_perspective import flip_action, flip_observation

BASE = ROOT / "output/ppo/RECEIVED_20261026"
PACKAGE = BASE / "PPO_TEST200_20261025"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_action(policy, observation):
    action = policy.select_move(observation)
    return action["action_id"] if isinstance(action, dict) else action


def load_policy(folder, key, checkpoint):
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(key, folder / "policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Policy(str(folder / checkpoint), device="cpu")


def main():
    bank = read(PACKAGE / "boards.json")
    old_root = ROOT / "output/dqn/UNSEEN10_FOUR_20261025"
    assert bank == read(old_root / "boards.json")
    assert (PACKAGE / "reproduce/xiangqi_lab/rules.py").read_text(encoding="utf-8") == (ROOT / "xiangqi_lab/rules.py").read_text(encoding="utf-8")
    plan = read(PACKAGE / "plan.json")
    expected_cases = [(b["id"], seed) for b in bank["boards"] for seed in b["opponent_seeds"]]
    boards = {b["id"]: b for b in bank["boards"]}
    audit = dict(status="running", training=False, rules_match=True, test_bank_matches=True,
                 source_archive_sha256=sha(BASE / "Xiangqi_PPO_Models_Test200_20261025.zip"),
                 torch_version=torch.__version__, models=[])
    jobs = [(key, ROOT / f"xiangqi_lab/models/local_ppo_rook20_seed{entry['seed']}", entry)
            for key, entry in plan["models"].items()]
    jobs.append(("old_double_dqn", ROOT / "xiangqi_lab/models/local_match_double_dqn", None))
    for key, folder, entry in jobs:
        is_ppo = entry is not None
        checkpoint = "weights.zip" if is_ppo else "best.pt"
        before_hash = sha(folder / checkpoint)
        expected_hash = entry["checkpoint_sha256"] if is_ppo else "b616b5c8fdb10587a8b9054484348ee0db118b02cf2b4e6e5e54cba9d90d7379"
        assert before_hash == expected_hash
        policy = load_policy(folder, key, checkpoint)
        network = policy.model if is_ppo else policy.policy.model
        before = {k: v.detach().clone() for k, v in network.state_dict().items()}
        csv_path = PACKAGE / "results" / f"{key}_games.csv" if is_ppo else old_root / "results/old_double_dqn_games.csv"
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
        assert [(r["board"], r["seed"]) for r in rows] == expected_cases
        counts, reasons, symmetry_checks, total_plies = Counter(), Counter(), 0, 0
        for index, row in enumerate(rows, 1):
            path = PACKAGE / row["replay"] if is_ppo else old_root / "episodes" / f"{key}_{row['board']}_{int(row['repeat']):02d}.json"
            if not path.exists() and not is_ppo:
                path = old_root / "results/episodes" / path.name
            episode = read(path)
            env = FixedOpponentEnv(160, "red")
            obs, info = env.reset(boards[row["board"]]["fen"], row["seed"])
            while info["outcome"] is None:
                action = get_action(policy, obs)
                if obs["ply"] % 20 == 0:
                    flipped_action = get_action(policy, flip_observation(obs))
                    assert flipped_action == flip_action(action), (key, "color symmetry")
                    symmetry_checks += 1
                obs, _, _, _, info = env.step(action)
            assert [m["move"] for m in env.game.history] == [m["move"] for m in episode["moves"]], (key, row["board"], row["repeat"])
            assert env.game.fen == row["final_fen"] and len(env.game.history) == int(row["plies"])
            outcome = env.game.outcome()
            assert outcome == episode["outcome"]
            result = "truncations" if outcome["truncated"] else "draws" if outcome["winner"] is None else "wins" if outcome["winner"] == "red" else "losses"
            expected_result = {"win": "wins", "loss": "losses", "draw": "draws", "truncation": "truncations"}.get(row["result"], row["result"])
            assert result == expected_result and outcome["reason"] == row["reason"]
            counts[result] += 1
            reasons[row["reason"]] += 1
            total_plies += len(env.game.history)
            if index % 50 == 0:
                print(f"{key}: {index}/200 games reproduced exactly", flush=True)
        assert sha(folder / checkpoint) == before_hash
        assert all(torch.equal(t, network.state_dict()[k]) for k, t in before.items())
        expected = read(PACKAGE / entry["summary"])["test"] if is_ppo else read(old_root / "results/comparison.json")["results"][key]
        for metric in ("wins", "losses", "draws", "truncations"):
            assert counts[metric] == expected[metric]
        result = dict(model=key, games=200, **{k: counts[k] for k in ("wins", "losses", "draws", "truncations")},
                      win_rate=counts["wins"] / 200, reasons=dict(reasons), legal_plies=total_plies,
                      checkpoint_sha256=before_hash, weights_unchanged=True, exact_trajectories=True,
                      color_symmetry_checks=symmetry_checks)
        audit["models"].append(result)
        (BASE / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
    audit.update(status="complete", passed=True, games=800, checked_utc=datetime.now(timezone.utc).isoformat())
    (BASE / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("All 800 games and color-view checks passed; no training.", flush=True)


if __name__ == "__main__":
    main()
