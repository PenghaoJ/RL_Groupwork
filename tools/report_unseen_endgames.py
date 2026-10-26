"""Verify saved test games and build a small offline report."""
import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
from rules import Game
from notebook_tools import board_svg, replay_episode
from evaluate_unseen_endgames import sha, write_json


def main(output):
    plan = json.loads((output / "plan.json").read_text())
    bank = json.loads((output / "boards.json").read_text())
    report = json.loads((output / "results/comparison.json").read_text())
    assert report["status"] == "complete" and report["training"] is False
    assert sha(output / "boards.json") == plan["board_bank_sha256"] == report["board_bank_sha256"]
    assert sha(output / "plan.json") == report["plan_sha256"]
    boards = {b["id"]: b for b in bank["boards"]}
    paired = {}
    for model in ["dqn", "double_dqn"]:
        with (output / f"results/{model}_games.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 200
        paired[model] = [(r["board"], r["seed"]) for r in rows]
        for index, row in enumerate(rows, 1):
            board = boards[row["board"]]
            assert row["seed"] == board["opponent_seeds"][int(row["repeat"]) - 1]
            file = output / "results/episodes" / f"{model}_{row['board']}_{int(row['repeat']):02d}.json"
            episode = json.loads(file.read_text())
            assert episode["initial_fen"] == board["fen"]
            assert episode["evaluation"]["checkpoint_sha256"] == plan["models"][model]["checkpoint_sha256"]
            assert episode["evaluation"]["seed"] == row["seed"]
            replay = replay_episode(episode)
            assert replay.fen == row["final_fen"] and len(replay.history) == int(row["plies"])
            assert replay.outcome() == episode["outcome"]
            result = replay.outcome()
            expected = ("truncations" if result["truncated"] else "draws" if result["winner"] is None
                        else "wins" if result["winner"] == "red" else "losses")
            assert row["result"] == expected and row["reason"] == result["reason"]
            if index % 50 == 0:
                print(f"Verified {model} replays: {index}/200", flush=True)
        actual = Counter(r["result"] for r in rows)
        for key in ["wins", "losses", "draws", "truncations"]:
            assert actual[key] == report["results"][model][key]
        for board_id in boards:
            selected = [r for r in rows if r["board"] == board_id]
            assert len(selected) == 20
            counts = Counter(r["result"] for r in selected)
            for key in ["wins", "losses", "draws", "truncations"]:
                assert counts[key] == report["results"][model]["per_board"][board_id][key]
        model_record = plan["models"][model]
        assert sha(output / model_record["backup"]) == sha(model_record["original_path"]) == model_record["file_sha256"]
    assert paired["dqn"] == paired["double_dqn"] and len(set(paired["dqn"])) == 200
    write_json(output / "verification.json", {
        "passed": True, "replayed_games": 400, "paired_cases": 200,
        "legal_moves_and_final_outcomes_verified": True, "csv_counts_verified": True,
        "original_and_backup_model_files_unchanged": True, "locked_board_bank_unchanged": True,
        "training": False, "report_sha256": sha(output / "results/comparison.json")})

    with (output / "board_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["board", "group", "fen", "dqn_wins", "dqn_losses", "dqn_draws", "dqn_limits",
                         "double_dqn_wins", "double_dqn_losses", "double_dqn_draws", "double_dqn_limits", "games_each"])
        for board in bank["boards"]:
            row = [board["id"], board["group"], board["fen"]]
            for name in ["dqn", "double_dqn"]:
                s = report["results"][name]["per_board"][board["id"]]
                row.extend(s[k] for k in ["wins", "losses", "draws", "truncations"])
            writer.writerow(row + [20])

    def score(s):
        return f"{s['wins']} / {s['losses']} / {s['draws']} / {s['truncations']}"

    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Ten unseen endgames - DQN vs Double DQN</title>',
             '<style>body{font:16px/1.6 system-ui,sans-serif;color:#203833;background:#f4f6f3;max-width:1040px;margin:auto;padding:24px}h1{line-height:1.2}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{text-align:left;border-bottom:1px solid #dbe2da;padding:12px}th{background:#e7eee7}section,details{background:white;border:1px solid #dbe2da;border-radius:12px;padding:20px;margin:16px 0}summary{font-weight:700;cursor:pointer}svg{display:block;width:100%;max-width:460px;height:auto;margin:16px auto}code{overflow-wrap:anywhere}a{color:#17685c}.note{border-left:4px solid #ba7a34;padding-left:16px}small{color:#566861}</style>',
             '<h1>Ten unseen endgames</h1><p>DQN vs Double DQN &middot; 25 October 2026</p>',
             '<p>10 fixed boards &times; 20 opponent seeds per board = <strong>200 games per model</strong>. Both saved best models play Red against random legal Black moves. No training, no exploration, and no weight updates.</p>',
             '<table><tr><th>Model</th><th>Best episode</th><th>Wins / Losses / Draws / Limits</th><th>Win rate</th></tr>']
    for name, label in [("dqn", "DQN"), ("double_dqn", "Double DQN")]:
        s = report["results"][name]
        parts.append(f'<tr><td>{label}</td><td>{s["best_episode"]}</td><td>{score(s)}</td><td>{s["win_rate"]:.1%}</td></tr>')
    parts += ['</table><p class="note">This is a small test on ten selected boards, not proof of general chess skill or of which algorithm is better overall. Repeated games on one board are not new boards. The opponent is random, not Pikafish. A 160-ply time limit is reported separately from a draw.</p>',
              '<h2>Results by piece type</h2><table><tr><th>Type</th><th>Boards</th><th>Games each model</th><th>DQN win rate</th><th>Double DQN win rate</th></tr>']
    for group, count in [("rook", 4), ("cannon", 3), ("horse", 3)]:
        d = report["results"]["dqn"]["per_group"][group]
        q = report["results"]["double_dqn"]["per_group"][group]
        parts.append(f'<tr><td>{group.title()} &amp; pawn</td><td>{count}</td><td>{d["games"]}</td><td>{d["win_rate"]:.1%}</td><td>{q["win_rate"]:.1%}</td></tr>')
    parts += ['</table><h2>The ten test boards</h2><p>Expand a board to see the position, FEN and saved games. Each game JSON can be loaded using the website\'s Load game control.</p>']
    for board in bank["boards"]:
        d = report["results"]["dqn"]["per_board"][board["id"]]
        q = report["results"]["double_dqn"]["per_board"][board["id"]]
        parts += [f'<details><summary>{board["id"]} &mdash; DQN {d["wins"]}/20 wins; Double DQN {q["wins"]}/20 wins</summary>',
                  f'<p>DQN W/L/D/limit: {score(d)}<br>Double DQN W/L/D/limit: {score(q)}</p>',
                  board_svg(Game(board["fen"]), board["id"]),
                  f'<p><strong>FEN</strong><br><code>{escape(board["fen"])}</code></p>',
                  f'<p><small>Why unseen: {escape(board["unseen_evidence"])}</small></p>']
        for name, label in [("dqn", "DQN"), ("double_dqn", "Double DQN")]:
            links = [f'<a href="results/episodes/{name}_{board["id"]}_{i:02d}.json" download>Game {i}</a>' for i in range(1, 21)]
            parts.append(f'<p>{label}: ' + ' &middot; '.join(links) + '</p>')
        parts.append('</details>')
    parts += ['<h2>Checks and files</h2><p>All 400 saved games replayed legally. Both original model packages and their backups are unchanged. The board set was locked before any model played it.</p>',
              '<p><a href="board_results.csv">Board results CSV</a> &middot; <a href="results/comparison.json">Full results</a> &middot; <a href="boards.json">Boards and generation moves</a> &middot; <a href="plan.json">Frozen test plan</a> &middot; <a href="verification.json">Verification</a></p>',
              '<p><small>The boards are valid custom endgames, not certified wins. They were generated by legal moves from valid custom positions; no full-game historical reachability proof is claimed. Each starts with fresh history. The same research draw rules are used as in training. This local run uses PyTorch ' + escape(report["torch_version"]) + ' for both models.</small></p></html>']
    (output / "report.html").write_text('\n'.join(parts), encoding="utf-8")
    repro = output / "reproduce"
    (repro / "tools").mkdir(parents=True, exist_ok=True)
    (repro / "xiangqi_lab").mkdir(exist_ok=True)
    for name in ["evaluate_unseen_endgames.py", "compare_rook_checkpoints.py", "report_unseen_endgames.py"]:
        shutil.copy2(ROOT / "tools" / name, repro / "tools" / name)
    for name in ["rules.py", "dqn_model.py", "dqn_training.py", "notebook_tools.py"]:
        shutil.copy2(ROOT / "xiangqi_lab" / name, repro / "xiangqi_lab" / name)
    (output / "START_HERE.txt").write_text(
        "Open report.html for the results and all ten boards.\n"
        "models/ contains untouched backups of the two best model packages.\n"
        "results/episodes/ contains all 400 website-compatible game records.\n"
        "boards.json records all FENs, generation moves and paired seeds.\n"
        "plan.json was frozen before testing. No training took place.\n"
        "The original model paths in plan.json are provenance only.\n"
        "To reproduce on another machine, install a CPU build of PyTorch, then run the scripts\n"
        "in reproduce/tools: prepare with --output NEW_FOLDER --dqn PATH_TO_DQN_ZIP\n"
        "--double-dqn PATH_TO_DDQN_ZIP, then evaluate --output NEW_FOLDER.\n"
        "Preparation uses no model scores. Compare generated board FENs and seeds with boards.json.\n"
        "Never train on this bank while treating it as a held-out test set.\n",
        encoding="utf-8")
    bundle = output.parent / "Xiangqi_Unseen10_Test_20261025.zip"
    with zipfile.ZipFile(bundle, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path(output.name) / path.relative_to(output)))
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
    print(f"Verified and packaged: {bundle} ({bundle.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
