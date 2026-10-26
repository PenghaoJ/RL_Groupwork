"""Freeze a ten-board holdout set, then compare two saved policies without learning."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
from rules import Game, PRESETS, coords, in_check, legal_moves, make_fen

SPLIT = "unseen-10-20261025-v1"
DQN_SHA = "b6ad0a3b60d4cde7846a887c33c0a60835710b617d9aaf0a131ea1b98cf8c037"
DDQN_SHA = "b616b5c8fdb10587a8b9054484348ee0db118b02cf2b4e6e5e54cba9d90d7379"
REPEATS = 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def material(game):
    return Counter(p for row in game.board for p in row if p)


def board_key(fen):
    return " ".join(fen.split()[:2])


def immediate_win(game):
    for move in game.legal:
        probe = Game(game.fen)
        outcome = probe.step(move)
        if outcome and outcome["winner"] == game.side:
            return True
    return False


def unseen_evidence(game, group):
    if group == "rook":
        # The only trained red pawn started on rank 4. It cannot move backwards.
        assert any(p == "P" and row == 6 for row, cells in enumerate(game.board) for p in cells)
        return "Red pawn on rank 3; training pawn started on rank 4 and cannot return to rank 3."
    piece = "C" if group == "cannon" else "H"
    assert material(game)[piece] == 1
    return f"Contains a red {piece}; this piece was absent from every training position."


def prepare(output, dqn, double_dqn):
    from compare_rook_checkpoints import load_best
    # Check identities before copying files or generating any test boards.
    candidates = {"dqn": (Path(dqn), load_best(dqn, "dqn", DQN_SHA)),
                  "double_dqn": (Path(double_dqn), load_best(double_dqn, "double_dqn", DDQN_SHA))}
    configs = [{k: v for k, v in vars(item[1][1]).items() if k != "algorithm"}
               for item in candidates.values()]
    assert configs[0] == configs[1]
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    models = {}
    for name, (source, (_, _, metadata)) in candidates.items():
        dest = output / "models" / source.name
        shutil.copy2(source, dest)
        assert sha(source) == sha(dest) == metadata["file_sha256"]
        models[name] = {**metadata, "original_path": str(source.resolve()),
                        "backup": str(dest.relative_to(output))}

    presets = {p["id"]: p["fen"] for p in PRESETS}
    rook = Game(presets["rook"])
    rook.board[5][2], rook.board[6][2] = "", "P"
    bases = {"rook": make_fen(rook.board, "red"),
             "cannon": presets["cannon"], "horse": presets["horse"]}
    used = {board_key(fen) for fen in presets.values()}
    boards = []
    for group, count in [("rook", 4), ("cannon", 3), ("horse", 3)]:
        for number in range(1, count + 1):
            board_id = f"{group}-{number:02d}"
            rng = random.Random(f"{SPLIT}:board:{board_id}")
            base = Game(bases[group])
            target_plies = 8 + number * 4
            for attempt in range(1, 501):
                game = Game(bases[group])
                for _ in range(target_plies):
                    if game.outcome() is not None:
                        break
                    choices = []
                    for move in game.legal:
                        r, c = coords(move[:2])
                        tr, tc = coords(move[2:])
                        if game.board[tr][tc]:
                            continue
                        if group == "rook" and game.board[r][c] == "P":
                            continue
                        choices.append(move)
                    if not choices:
                        break
                    game.step(rng.choice(choices), actor="board-builder")
                fen = make_fen(game.board, "red", 0, 1)
                if len(game.history) != target_plies or game.outcome() is not None:
                    continue
                if in_check(game.board, "red") or in_check(game.board, "black"):
                    continue
                fresh = Game(fen)
                mirrored = make_fen([list(reversed(row)) for row in fresh.board], "red")
                difference = sum(a != b for ra, rb in zip(base.board, fresh.board) for a, b in zip(ra, rb))
                if (board_key(fen) in used or board_key(mirrored) in used or difference < 4
                        or len(fresh.legal) < 6 or len(legal_moves(fresh.board, "black")) < 3
                        or immediate_win(fresh)):
                    continue
                assert material(fresh) == material(base)
                entry = {"id": board_id, "group": group, "fen": fen,
                         "piece_count": sum(material(fresh).values()),
                         "red_legal_moves": len(fresh.legal),
                         "black_legal_moves": len(legal_moves(fresh.board, "black")),
                         "unseen_evidence": unseen_evidence(fresh, group),
                         "generation": {"base_fen": bases[group], "attempt": attempt,
                                        "legal_moves": [h["move"] for h in game.history],
                                        "seed": f"{SPLIT}:board:{board_id}"},
                         "opponent_seeds": [f"{SPLIT}:opponent:{board_id}:{i:02d}" for i in range(REPEATS)]}
                boards.append(entry)
                used.add(board_key(fen))
                print(f"Prepared {board_id}: {fen}", flush=True)
                break
            else:
                raise RuntimeError(f"No valid board found for {board_id}")
    bank = {"split": SPLIT, "created_utc": datetime.now(timezone.utc).isoformat(),
            "board_count": len(boards), "games_per_board_per_model": REPEATS,
            "boards": boards,
            "selection": "Deterministic legal non-capture rollouts; no model scores used in board selection.",
            "checks": ["Valid piece counts and locations", "Neither side starts in check",
                       "At least 6 Red and 3 Black legal moves", "No immediate Red win",
                       "Unique boards and no horizontal-mirror duplicates", "Not any website preset",
                       "Unseen even among states reachable from the training start"],
            "limits": "Custom legal endgames, not certified wins or a representative random sample of all Xiangqi endgames. Legal rollouts start from custom valid positions, not from a recorded full game. History and counters reset for each puzzle."}
    assert len(boards) == 10
    write_json(output / "boards.json", bank)
    plan = {"status": "locked_before_test", "created_utc": bank["created_utc"],
            "split": SPLIT, "board_bank_sha256": sha(output / "boards.json"),
            "games_per_model": 10 * REPEATS, "models": models,
            "learner": "red", "opponent": "uniform random legal moves",
            "max_plies": 160, "epsilon": 0, "training": False,
            "device": "cpu", "seed_pairing": "Same seed per board and repeat for both models; legal replies may differ after different policy moves.",
            "rules": "Existing xiangqi-research-v1, unchanged; threefold or 120 non-capture plies draw; 160 ply limit reported separately."}
    write_json(output / "plan.json", plan)
    print("Board bank and model backups locked. No test games have run.", flush=True)


def evaluate(output):
    import torch
    from compare_rook_checkpoints import load_best
    from dqn_model import encode_state, greedy_action
    from dqn_training import FixedOpponentEnv
    torch.set_num_threads(2)
    plan_path = output / "plan.json"
    plan_sha = sha(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert sha(output / "boards.json") == plan["board_bank_sha256"]
    bank = json.loads((output / "boards.json").read_text(encoding="utf-8"))
    results_dir = output / "results"
    results_dir.mkdir(exist_ok=False)
    (results_dir / "episodes").mkdir()
    report = {"status": "running", "split": SPLIT, "training": False, "epsilon": 0,
              "torch_version": str(torch.__version__), "python_version": sys.version,
              "device": "cpu", "plan_sha256": plan_sha,
              "board_bank_sha256": plan["board_bank_sha256"], "results": {}}
    write_json(results_dir / "comparison.json", report)
    all_rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for name, expected in [("dqn", DQN_SHA), ("double_dqn", DDQN_SHA)]:
            metadata = plan["models"][name]
            checkpoint = output / metadata["backup"]
            assert sha(checkpoint) == metadata["file_sha256"]
            model, config, loaded = load_best(checkpoint, name, expected)
            assert config.learner == plan["learner"] and config.max_plies == plan["max_plies"]
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            before = {k: v.clone() for k, v in model.state_dict().items()}
            rows = []
            with (results_dir / f"{name}_games.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["model", "board", "group", "repeat", "seed", "result", "reason", "plies", "final_fen"])
                writer.writeheader()
                for board in bank["boards"]:
                    for repeat, seed in enumerate(board["opponent_seeds"], 1):
                        env = FixedOpponentEnv(plan["max_plies"], plan["learner"])
                        observation, info = env.reset(board["fen"], seed)
                        assert info["outcome"] is None
                        while info["outcome"] is None:
                            action = greedy_action(model, encode_state(observation), observation["legal_action_ids"], "cpu")
                            observation, _, _, _, info = env.step(action)
                        outcome = info["outcome"]
                        result = ("truncations" if outcome["truncated"] else "draws" if outcome["winner"] is None
                                  else "wins" if outcome["winner"] == "red" else "losses")
                        row = {"model": name, "board": board["id"], "group": board["group"],
                               "repeat": repeat, "seed": seed, "result": result,
                               "reason": outcome["reason"], "plies": len(env.game.history), "final_fen": env.game.fen}
                        writer.writerow(row)
                        handle.flush()
                        rows.append(row)
                        episode = env.game.export()
                        episode["evaluation"] = {"model": name, "checkpoint_sha256": expected,
                                                 "board": board["id"], "seed": seed, "training": False}
                        write_json(results_dir / "episodes" / f"{name}_{board['id']}_{repeat:02d}.json", episode)
                    counts = Counter(r["result"] for r in rows if r["board"] == board["id"])
                    print(f"{name} {board['id']}: {dict(counts)} ({len(rows)}/200)", flush=True)
                    write_json(output / "progress.json", {"status": "testing", "model": name,
                               "games_completed_this_model": len(rows), "seconds": time.perf_counter() - started})
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
            assert sha(checkpoint) == loaded["file_sha256"] == metadata["file_sha256"]
            assert sha(metadata["original_path"]) == metadata["file_sha256"]
            all_rows.extend(rows)
            def stats(selected):
                counts = Counter(r["result"] for r in selected)
                return {**{k: counts[k] for k in ["wins", "losses", "draws", "truncations"]},
                        "games": len(selected), "win_rate": counts["wins"] / len(selected)}
            report["results"][name] = {**stats(rows), "best_episode": loaded["best_episode"],
                                       "checkpoint_sha256": expected, "weights_unchanged": True,
                                       "per_group": {g: stats([r for r in rows if r["group"] == g]) for g in ["rook", "cannon", "horse"]},
                                       "per_board": {b["id"]: stats([r for r in rows if r["board"] == b["id"]]) for b in bank["boards"]}}
            write_json(results_dir / "comparison.json", report)
    pairs = {}
    for row in all_rows:
        pairs.setdefault((row["board"], row["seed"]), {})[row["model"]] = row["result"]
    assert len(pairs) == 200 and all(set(pair) == {"dqn", "double_dqn"} for pair in pairs.values())
    assert sha(plan_path) == plan_sha and sha(output / "boards.json") == plan["board_bank_sha256"]
    report.update(status="complete", seconds=time.perf_counter() - started,
                  paired_seed_check=True, board_bank_unchanged=True,
                  scope="10 preselected unseen boards, 20 opponent seeds each; one checkpoint per algorithm, not a general algorithm ranking.")
    write_json(results_dir / "comparison.json", report)
    write_json(output / "progress.json", {"status": "complete", "games_per_model": 200})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "evaluate"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dqn")
    parser.add_argument("--double-dqn")
    args = parser.parse_args()
    if args.mode == "prepare":
        if not args.dqn or not args.double_dqn:
            parser.error("prepare requires --dqn and --double-dqn")
        prepare(args.output, args.dqn, args.double_dqn)
    else:
        evaluate(args.output)
