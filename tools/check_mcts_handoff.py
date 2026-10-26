"""Check an extracted handoff without training or changing model weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
from board_bank import BoardBank, position_key, mirror_key
from engines import check_model, close_workers
from rules import Game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-opponents", action="store_true")
    args = parser.parse_args()
    count = 0
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        count += 1
    bank = BoardBank(json.loads((ROOT / "experiments/rook20_20261025/boards.json").read_text(encoding="utf-8")))
    assert len(bank.training) == 20 and len(bank.validation) == 6
    tests = json.loads((ROOT / "output/dqn/UNSEEN10_FOUR_20261025/boards.json").read_text(encoding="utf-8"))["boards"]
    forbidden = {key(b["fen"]) for b in tests for key in (position_key, mirror_key)}
    for board in bank.training + bank.validation:
        assert position_key(board["fen"]) not in forbidden
        game = Game(board["fen"])
        before = game.export()
        clone = copy.deepcopy(game)
        clone.step(clone.legal[0])
        assert game.export() == before
    notebook = json.loads((ROOT / "notebooks/MCTS_Handoff_Start.ipynb").read_text(encoding="utf-8"))
    context = {"__name__": "__handoff_notebook__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), "MCTS_Handoff_Start.ipynb", "exec"), context)
    models = ["custom-team-template"]
    if args.with_opponents:
        models += ["custom-local-match-double-dqn", "custom-ppo-friend-rook20-seed42"]
    checks = []
    try:
        for model in models:
            checks.append(check_model(model))
            print(f"PASS {model}: 3 legal positions", flush=True)
    finally:
        close_workers()
    print(json.dumps({"passed": True, "files_verified": count, "training_starts": 20,
                      "validation_starts": 6, "test_starts": len(tests), "starter_notebook": "passed",
                      "training": False, "models": checks}, indent=2))


if __name__ == "__main__":
    main()
