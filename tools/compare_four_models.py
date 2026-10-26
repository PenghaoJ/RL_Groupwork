"""Compare four frozen best policies on the existing ten-board test bank."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
from html import escape
import io
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
import torch
from dqn_model import MODEL_SCHEMA, QNetwork, encode_state, greedy_action
from dqn_training import FixedOpponentEnv
from notebook_tools import replay_episode

MODELS = {
    "old_dqn": ("Old DQN", "dqn", "FAIR_COMPARISON_20261022/models/DQN_R2_REPRO_20261022_S42_BEST_EP0100_FINAL.zip",
                "b6ad0a3b60d4cde7846a887c33c0a60835710b617d9aaf0a131ea1b98cf8c037", 100),
    "new_dqn": ("New DQN (20 boards)", "dqn", "ROOK20_20261025_S42/models/DQN_ROOK20_20261025_S42_BEST_EP0100.zip",
                "7af30e2b6bc4ad88b7d2c8fd89c44f8130daa1d22be8bb90be7618bc1b25c219", 100),
    "old_double_dqn": ("Old Double DQN", "double_dqn", "FAIR_COMPARISON_20261022/models/DoubleDQN_Original_Rook_20261022_S42_BestEp0200.zip",
                       "b616b5c8fdb10587a8b9054484348ee0db118b02cf2b4e6e5e54cba9d90d7379", 200),
    "new_double_dqn": ("New Double DQN (20 boards)", "double_dqn", "ROOK20_20261025_S42/models/DoubleDQN_ROOK20_20261025_S42_BEST_EP0100.zip",
                       "f99cdf676085cbdc8d367d315ff2b5a0c63a9b0aa2171292d1f07cd576887bd6", 100),
}
RESULT_KEYS = ("wins", "losses", "draws", "truncations")
GROUPS = ("rook", "cannon", "horse")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sources():
    return {name: hashlib.sha256((ROOT / "xiangqi_lab" / name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in ("rules.py", "dqn_model.py", "dqn_training.py", "notebook_tools.py")}


def load_policy(path, expected):
    with zipfile.ZipFile(path) as archive:
        entries = [n for n in archive.namelist() if n.endswith("/best.pt")]
        assert len(entries) == 1
        raw = archive.read(entries[0])
    assert hashlib.sha256(raw).hexdigest() == expected["checkpoint_sha256"]
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    assert saved["schema"] == MODEL_SCHEMA and saved["kind"] == "policy"
    assert saved["episode"] == expected["best_episode"]
    assert saved["config"]["algorithm"] == expected["algorithm"]
    assert saved["config"]["learner"] == "red" and saved["config"]["max_plies"] == 160
    for name in ("rules.py", "dqn_model.py"):
        assert saved["source_hashes"][name] == sources()[name]
    model = QNetwork(**saved["network_config"])
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    return model, saved


def prepare(output):
    original = ROOT / "output/dqn/UNSEEN10_20261025"
    old_plan = json.loads((original / "plan.json").read_text())
    bank = json.loads((original / "boards.json").read_text())
    assert sha(original / "boards.json") == old_plan["board_bank_sha256"]
    assert len(bank["boards"]) == 10
    assert all(len(b["opponent_seeds"]) == 20 for b in bank["boards"])
    cases = [(b["id"], s) for b in bank["boards"] for s in b["opponent_seeds"]]
    assert len(set(cases)) == 200
    new_bank = json.loads((ROOT / "experiments/rook20_20261025/boards.json").read_text())
    test_positions = {" ".join(b["fen"].split()[:2]) for b in bank["boards"]}
    assert not test_positions.intersection(" ".join(b["fen"].split()[:2])
                                          for b in new_bank["training"] + new_bank["validation"])
    prepared = {}
    for key, (label, algorithm, relative, checkpoint_sha, episode) in MODELS.items():
        source = ROOT / "output/dqn" / relative
        expected = dict(label=label, algorithm=algorithm, checkpoint_sha256=checkpoint_sha,
                        best_episode=episode, original_path=str(source.resolve()),
                        backup=f"models/{source.name}", file_sha256=sha(source))
        _, saved = load_policy(source, expected)
        expected["training_config"] = saved["config"]
        prepared[key] = expected
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    for expected in prepared.values():
        shutil.copy2(expected["original_path"], output / expected["backup"])
    shutil.copy2(original / "boards.json", output / "boards.json")
    shutil.copy2(original / "results/comparison.json", output / "previous_results.json")
    for old, prior_key in (("old_dqn", "dqn"), ("old_double_dqn", "double_dqn")):
        shutil.copy2(original / f"results/{prior_key}_games.csv", output / f"previous_{old}_games.csv")
    plan = dict(status="locked_before_test", created_utc=datetime.now(timezone.utc).isoformat(),
                split=bank["split"], board_bank_sha256=sha(output / "boards.json"),
                seed_bank="Exactly the original 200 board/seed pairs, reused for all four models",
                games_per_model=200, total_games=800, training=False, epsilon=0,
                learner="red", opponent="uniform-legal-random-v1", device="cpu", max_plies=160,
                rules="Existing research rules: threefold or 120 no-capture plies draw; 160-ply limit separate",
                sources=sources(), models=prepared,
                scope="Previously inspected test bank reused for a regression comparison; not a fresh blind holdout. No test-based checkpoint selection.")
    write_json(output / "plan.json", plan)
    print("Four best models and original 200 test cases locked. No training.", flush=True)


def stats(rows):
    counts = Counter(row["result"] for row in rows)
    return {**{key: counts[key] for key in RESULT_KEYS}, "games": len(rows),
            "win_rate": counts["wins"] / len(rows),
            "reasons": dict(Counter(row["reason"] for row in rows))}


@torch.inference_mode()
def evaluate(output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((output / "plan.json").read_text())
    plan_sha = sha(output / "plan.json")
    assert sources() == plan["sources"]
    assert sha(output / "boards.json") == plan["board_bank_sha256"]
    bank = json.loads((output / "boards.json").read_text())
    folder = output / "results"
    folder.mkdir(exist_ok=False)
    (folder / "episodes").mkdir()
    report = dict(status="running", training=False, epsilon=0, device="cpu", results={},
                  torch_version=str(torch.__version__), python_version=sys.version,
                  plan_sha256=plan_sha, board_bank_sha256=plan["board_bank_sha256"])
    all_rows = {}
    started = time.perf_counter()
    for key, expected in plan["models"].items():
        path = output / expected["backup"]
        assert sha(path) == expected["file_sha256"]
        model, saved = load_policy(path, expected)
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rows = []
        fields = ["model", "board", "group", "repeat", "seed", "result", "reason", "plies", "final_fen"]
        with (folder / f"{key}_games.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for board in bank["boards"]:
                for repeat, seed in enumerate(board["opponent_seeds"], 1):
                    env = FixedOpponentEnv(plan["max_plies"], plan["learner"])
                    obs, info = env.reset(board["fen"], seed)
                    assert info["outcome"] is None
                    while info["outcome"] is None:
                        action = greedy_action(model, encode_state(obs), obs["legal_action_ids"], "cpu")
                        obs, _, _, _, info = env.step(action)
                    outcome = info["outcome"]
                    result = ("truncations" if outcome["truncated"] else "draws" if outcome["winner"] is None
                              else "wins" if outcome["winner"] == "red" else "losses")
                    row = dict(model=key, board=board["id"], group=board["group"], repeat=repeat,
                               seed=seed, result=result, reason=outcome["reason"],
                               plies=len(env.game.history), final_fen=env.game.fen)
                    writer.writerow(row)
                    rows.append(row)
                    episode = env.game.export()
                    episode["evaluation"] = dict(model=key, checkpoint_sha256=expected["checkpoint_sha256"],
                                                 board=board["id"], seed=seed, training=False, epsilon=0)
                    write_json(folder / "episodes" / f"{key}_{board['id']}_{repeat:02d}.json", episode)
                stream.flush()
                print(f"{key} {board['id']}: {stats(rows)['wins']} wins so far ({len(rows)}/200)", flush=True)
                write_json(output / "progress.json", dict(status="testing", model=key,
                           games_this_model=len(rows), total_games=sum(map(len, all_rows.values()))+len(rows)))
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        assert sha(path) == sha(expected["original_path"]) == expected["file_sha256"]
        all_rows[key] = rows
        report["results"][key] = {**stats(rows), "label": expected["label"],
                                  "best_episode": saved["episode"], "weights_unchanged": True,
                                  "checkpoint_sha256": expected["checkpoint_sha256"],
                                  "per_group": {g: stats([r for r in rows if r["group"] == g]) for g in GROUPS},
                                  "per_board": {b["id"]: stats([r for r in rows if r["board"] == b["id"]]) for b in bank["boards"]}}
        write_json(folder / "comparison.json", report)
    paired_cases = [[(r["board"], r["seed"]) for r in rows] for rows in all_rows.values()]
    assert all(cases == paired_cases[0] for cases in paired_cases) and len(set(paired_cases[0])) == 200
    old_matches = {}
    for key in ("old_dqn", "old_double_dqn"):
        with (output / f"previous_{key}_games.csv").open(newline="", encoding="utf-8") as stream:
            previous = list(csv.DictReader(stream))
        fields = ("board", "seed", "result", "reason", "plies", "final_fen")
        old_matches[key] = all(all(str(a[k]) == str(b[k]) for k in fields)
                               for a, b in zip(previous, all_rows[key])) and len(previous) == 200
        assert old_matches[key], "Old-model rerun differs from earlier results; investigate before interpreting."
    comparisons = {}
    for algorithm in ("dqn", "double_dqn"):
        old, new = report["results"][f"old_{algorithm}"], report["results"][f"new_{algorithm}"]
        pairs = list(zip(all_rows[f"old_{algorithm}"], all_rows[f"new_{algorithm}"]))
        comparisons[algorithm] = dict(win_change=new["wins"]-old["wins"],
            win_rate_change_pp=100*(new["win_rate"]-old["win_rate"]),
            new_win_old_not=sum(a["result"]!="wins" and b["result"]=="wins" for a,b in pairs),
            old_win_new_not=sum(a["result"]=="wins" and b["result"]!="wins" for a,b in pairs),
            per_group_change_pp={g:100*(new["per_group"][g]["win_rate"]-old["per_group"][g]["win_rate"]) for g in GROUPS})
    assert sha(output / "plan.json") == plan_sha
    assert sha(output / "boards.json") == plan["board_bank_sha256"]
    report.update(status="complete", seconds=time.perf_counter()-started, paired_seed_check=True,
                  old_results_reproduced_exactly=old_matches, comparisons=comparisons,
                  scope=plan["scope"])
    write_json(folder / "comparison.json", report)
    write_json(output / "progress.json", dict(status="complete", games_per_model=200, total_games=800))
    print(json.dumps({k:{s:v[s] for s in (*RESULT_KEYS,"win_rate")} for k,v in report["results"].items()},indent=2), flush=True)


def verify_and_report(output):
    plan = json.loads((output / "plan.json").read_text())
    report = json.loads((output / "results/comparison.json").read_text())
    bank = json.loads((output / "boards.json").read_text())
    assert report["status"] == "complete" and report["training"] is False
    boards = {b["id"]: b for b in bank["boards"]}
    verified = 0
    for key in plan["models"]:
        with (output / f"results/{key}_games.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 200
        for row in rows:
            episode = json.loads((output / "results/episodes" / f"{key}_{row['board']}_{int(row['repeat']):02d}.json").read_text())
            assert episode["initial_fen"] == boards[row["board"]]["fen"]
            assert row["seed"] == boards[row["board"]]["opponent_seeds"][int(row["repeat"])-1]
            assert episode["evaluation"]["seed"] == row["seed"]
            assert episode["evaluation"]["checkpoint_sha256"] == plan["models"][key]["checkpoint_sha256"]
            replay = replay_episode(episode)
            assert replay.fen == row["final_fen"] and len(replay.history) == int(row["plies"])
            assert replay.outcome() == episode["outcome"]
            outcome = replay.outcome()
            result = ("truncations" if outcome["truncated"] else "draws" if outcome["winner"] is None
                      else "wins" if outcome["winner"] == "red" else "losses")
            assert row["result"] == result and row["reason"] == outcome["reason"]
            verified += 1
        actual = stats(rows)
        assert all(actual[k] == report["results"][key][k] for k in (*RESULT_KEYS,"win_rate"))
        for board in boards:
            selected = [r for r in rows if r["board"] == board]
            assert len(selected) == 20 and stats(selected) == report["results"][key]["per_board"][board]
        print(f"Verified {key}: 200 legal game replays", flush=True)
    for model in plan["models"].values():
        assert sha(model["original_path"]) == sha(output/model["backup"]) == model["file_sha256"]
    assert sha(output/"boards.json") == report["board_bank_sha256"]
    assert sha(output/"plan.json") == report["plan_sha256"]
    write_json(output / "verification.json", dict(passed=True, replayed_games=verified,
               paired_cases=200, original_and_backup_weights_unchanged=True, training=False,
               report_sha256=sha(output / "results/comparison.json")))
    with (output / "board_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["board","group","fen","model","wins","losses","draws","truncations","games","win_rate"])
        for board in bank["boards"]:
            for key in plan["models"]:
                s = report["results"][key]["per_board"][board["id"]]
                writer.writerow([board["id"],board["group"],board["fen"],key]+[s[k] for k in (*RESULT_KEYS,"games","win_rate")])
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Four models: the same ten test boards</title><style>body{font:16px/1.6 system-ui;color:#203833;background:#f4f6f3;max-width:1100px;margin:auto;padding:24px}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{text-align:left;border-bottom:1px solid #dbe2da;padding:10px}th{background:#e7eee7}code{overflow-wrap:anywhere}a{color:#17685c}details{margin:12px 0;padding:14px;background:white}.scroll{overflow:auto}</style>',
             '<h1>Four models: the same ten test boards</h1><p>25 October 2026. 10 boards &times; 20 fixed seeds = 200 games per model, 800 games total. All models play Red against random legal Black moves. No training, no exploration, no weight updates.</p>',
             '<p>Old: trained on one rook start. New: trained from scratch on 20 rook starts. Each run completed 1,500 episodes; these tests use its saved best checkpoint.</p>',
             '<div class="scroll"><table><tr><th>Model</th><th>Best episode</th><th>Wins</th><th>Losses</th><th>Draws</th><th>Limits</th><th>Win rate</th></tr>']
    for key, s in report["results"].items():
        parts.append(f'<tr><td>{escape(s["label"])}</td><td>{s["best_episode"]}</td>'+''.join(f'<td>{s[k]}</td>' for k in RESULT_KEYS)+f'<td>{s["win_rate"]:.1%}</td></tr>')
    parts.append('</table></div><h2>Change from old to new</h2>')
    for algorithm, values in report["comparisons"].items():
        parts.append(f'<p>{escape(algorithm)}: {values["win_change"]:+d} wins / 200; {values["win_rate_change_pp"]:+.1f} percentage points.</p>')
    parts.append('<h2>Results by piece type</h2><div class="scroll"><table><tr><th>Type</th><th>Games per model</th>'+''.join(f'<th>{escape(m["label"])}</th>' for m in plan["models"].values())+'</tr>')
    for group, games in (("rook",80),("cannon",60),("horse",60)):
        parts.append(f'<tr><td>{group.title()} &amp; pawn</td><td>{games}</td>'+''.join(f'<td>{s["per_group"][group]["wins"]}/{games} ({s["per_group"][group]["win_rate"]:.1%})</td>' for s in report["results"].values())+'</tr>')
    parts.append('</table></div><h2>Boards and saved games</h2>')
    for board in bank["boards"]:
        parts.append(f'<details><summary>{board["id"]}</summary><p><code>{escape(board["fen"])}</code></p>')
        for key, metadata in plan["models"].items():
            wins = report["results"][key]["per_board"][board["id"]]["wins"]
            links = ' &middot; '.join(f'<a href="results/episodes/{key}_{board["id"]}_{i:02d}.json" download>{i}</a>' for i in range(1,21))
            parts.append(f'<p>{escape(metadata["label"])}: {wins}/20 wins<br>{links}</p>')
        parts.append('</details>')
    parts.append('<h2>What this test means</h2><p>These are ten previously inspected test boards, not a fresh blind sample. They were not used for training or checkpoint selection. Repeated seeds on the same board are not independent new boards. This compares four specific saved models, not all DQN and Double DQN models. The opponent is random, not Pikafish. Valid custom positions are not guaranteed wins.</p><p>All 800 saved games replayed legally. Model files and in-memory weights stayed unchanged. Both old models exactly reproduced their previous games. Limits are separate from draws. Same seeds do not guarantee identical opponent moves after different model moves.</p><p><a href="board_results.csv">Board results CSV</a> &middot; <a href="results/comparison.json">Full results</a> &middot; <a href="verification.json">Checks</a> &middot; <a href="boards.json">Boards and seeds</a> &middot; <a href="plan.json">Locked plan</a></p></html>')
    (output / "report.html").write_text('\n'.join(parts), encoding="utf-8")
    repro = output / "reproduce"
    (repro/"tools").mkdir(parents=True,exist_ok=True)
    (repro/"xiangqi_lab").mkdir(exist_ok=True)
    shutil.copy2(__file__,repro/"tools/compare_four_models.py")
    for source in ("rules.py","dqn_model.py","dqn_training.py","notebook_tools.py"):
        shutil.copy2(ROOT/"xiangqi_lab"/source,repro/"xiangqi_lab"/source)
    (output / "START_HERE.txt").write_text(
        "Open report.html. models/ has four untouched best policy packages. results/episodes/ has 800 website-compatible game records.\n"
        "Training was disabled. boards.json is the original ten-board bank, including the same 200 seeds.\n"
        "To rerun: use the copied Python sources and PyTorch, copy plan.json, boards.json, previous_*.csv, and models/ into a fresh output folder, then run compare_four_models.py evaluate --output FOLDER. Original absolute model paths in plan.json must be updated to the preserved ZIP paths on your computer before running.\n",encoding="utf-8")
    bundle = output.parent / "Xiangqi_Four_Models_Test200_20261025.zip"
    with zipfile.ZipFile(bundle,"x",zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path,Path(output.name)/path.relative_to(output))
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
    print(f"Verified {verified} games. Saved {bundle}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","evaluate","report"])
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    {"prepare":prepare,"evaluate":evaluate,"report":verify_and_report}[args.mode](args.output)
