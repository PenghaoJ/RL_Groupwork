"""Create 20 training and 6 validation rook positions before running either model."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
from board_bank import BoardBank, position_key, mirror_key
from rules import Game, PRESETS, coords, in_check, legal_moves, make_fen

NAME = "ROOK20_20261025_S42"
DEST = ROOT / "experiments" / "rook20_20261025"
OUTPUT = ROOT / "output/dqn" / NAME


def immediate_win(game):
    for move in game.legal:
        probe = Game(game.fen)
        result = probe.step(move)
        if result and result["winner"] == "red":
            return True
    return False


def main():
    previous = json.loads((ROOT / "output/dqn/UNSEEN10_20261025/boards.json").read_text())
    excluded = [b["fen"] for b in previous["boards"]] + [p["fen"] for p in PRESETS]
    data = {"schema": "xiangqi-lab.board-bank.v1", "id": NAME,
            "training": [], "validation": [], "excluded_fens": excluded,
            "selection": "Fixed seeded legal non-capture rollouts; no model scores used.",
            "schedule": "Each block of 20 training episodes visits each training start once in shuffled order.",
            "limits": "Valid custom endgames; not certified winning positions or a representative sample of all endgames. Independence refers to starting-board splits, not all reachable states.",
            "old_test_protection": "Rook test starts have a red pawn on rank 3. All new starts have it on rank 4 or higher, and it cannot move backwards. Cannon and horse test material is absent."}
    seen = {key for fen in excluded for key in [position_key(fen), mirror_key(fen)]}
    for split, count in [("training", 20), ("validation", 6)]:
        for number in range(1, count + 1):
            identifier = f"rook20-{split}-{number:02d}"
            rng = random.Random(f"{NAME}:{split}:{number}")
            base = Game(PRESETS[0]["fen"])
            base.board[5][2] = ""
            base.board[5][2 * ((number - 1) % 5)] = "P"
            base.board[4][4] = ""
            base.board[3 + (number % 2)][2 * ((number + 1) % 5)] = "p"
            base_fen = make_fen(base.board, "red")
            Game(base_fen)
            plies = 8 + 4 * ((number - 1) % 5)
            for attempt in range(1, 1001):
                game = Game(base_fen)
                for _ in range(plies):
                    if game.outcome() is not None:
                        break
                    options = []
                    for move in game.legal:
                        r, c = coords(move[:2]); tr, tc = coords(move[2:])
                        if game.board[tr][tc]:
                            continue
                        if game.board[r][c] == "P" and tr < 2:
                            continue
                        if game.board[r][c] == "p" and tr > 6:
                            continue
                        options.append(move)
                    if not options:
                        break
                    game.step(rng.choice(options), actor="board-builder")
                if len(game.history) != plies or game.outcome() is not None:
                    continue
                if in_check(game.board, "red") or in_check(game.board, "black"):
                    continue
                fen = make_fen(game.board, "red")
                fresh = Game(fen)
                changed = sum(a != b for ra, rb in zip(base.board, fresh.board) for a, b in zip(ra, rb))
                if (position_key(fen) in seen or changed < 4 or len(fresh.legal) < 6
                        or len(legal_moves(fresh.board, "black")) < 3 or immediate_win(fresh)):
                    continue
                assert Counter(p for row in fresh.board for p in row if p) == Counter("KR Pkap".replace(" ", ""))
                assert all(r <= 5 for r, row in enumerate(fresh.board) for p in row if p == "P")
                data[split].append({"id": identifier, "fen": fen,
                                    "generation": {"base_fen": base_fen, "attempt": attempt,
                                                   "moves": [h["move"] for h in game.history]},
                                    "red_legal_moves": len(fresh.legal),
                                    "black_legal_moves": len(legal_moves(fresh.board, "black"))})
                seen.update([position_key(fen), mirror_key(fen)])
                print(identifier, fen, flush=True)
                break
            else:
                raise RuntimeError("Could not construct " + identifier)
    bank = BoardBank(data)
    DEST.mkdir(parents=True, exist_ok=False)
    (DEST / "boards.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    backups = OUTPUT / "baselines"
    backups.mkdir()
    original = ROOT / "output/dqn/FAIR_COMPARISON_20261022"
    manifest = json.loads((original / "model-manifest.json").read_text())
    for model in manifest.values():
        source = original / "models" / model["file"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == model["zip_sha256"]
        with zipfile.ZipFile(source) as z:
            checkpoint = next(n for n in z.namelist() if n.endswith("/best.pt"))
            assert hashlib.sha256(z.read(checkpoint)).hexdigest() == model["checkpoint_sha256"]
        shutil.copy2(source, backups / source.name)
    (backups / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plan = {"experiment": NAME, "board_bank_sha256": bank.sha256, "training_boards": 20,
            "validation_boards": 6, "excluded_old_test_boards": 10,
            "initialization": "Both algorithms start from the same random initialization, seed 42; old checkpoints are not resumed.",
            "min_episodes": 1000, "max_episodes": 3000, "eval_every": 100, "eval_games": 60,
            "validation_games_per_board": 10, "patience": 5, "min_delta": 0.02,
            "epsilon_decay_steps": 100000, "max_plies": 160,
            "model_names": ["DQN_ROOK20_20261025_S42", "DoubleDQN_ROOK20_20261025_S42"],
            "status": "prepared; no formal training yet"}
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print("Prepared:", bank.sha256, flush=True)


if __name__ == "__main__":
    main()
