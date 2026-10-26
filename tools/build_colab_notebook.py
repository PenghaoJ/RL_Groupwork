"""Build the standalone Colab environment notebook from the actual app sources.

Run again after changing rules.py, training.py or notebook_tools.py.
No engine, credentials, runtime, model weights or browser server are bundled.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "DQN_ChineseChess.ipynb"
SOURCES = ("rules.py", "training.py", "notebook_tools.py", "dqn_model.py", "dqn_training.py")


def build_notebook():
    sources = {name: (ROOT / "xiangqi_lab" / name).read_text(encoding="utf-8") for name in SOURCES}
    hashes = {name: hashlib.sha256(source.encode("utf-8")).hexdigest() for name, source in sources.items()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    cells = []

    def add(kind, key, source, **metadata):
        cell = {"cell_type": kind, "metadata": {"id": key, **metadata},
                "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(key, source):
        add("markdown", key, source)

    def code(key, source, **metadata):
        add("code", key, source, **metadata)

    md("intro", """
        # Xiangqi Lab — DQN

        Set up the board rules, train DQN on a fixed endgame, and export a model for the website.
    """)
    md("setup-note", """
        ## 1. Load the project rules

        Loads the project rules and checks the included source files.
    """)
    bootstrap = f'''# @title Load the shared environment (run this cell first)
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import tempfile

SOURCE_FILES = json.loads({json.dumps(sources, ensure_ascii=False)!r})
SOURCE_HASHES = {hashes!r}
SOURCE_BASE_COMMIT = {commit!r}
SNAPSHOT_ID = hashlib.sha256(json.dumps(SOURCE_HASHES, sort_keys=True).encode()).hexdigest()[:16]
SOURCE_DIR = Path(tempfile.gettempdir()) / "xiangqi_lab_colab" / SNAPSHOT_ID
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
for name, source in SOURCE_FILES.items():
    data = source.encode("utf-8")
    if hashlib.sha256(data).hexdigest() != SOURCE_HASHES[name]:
        raise ValueError("Source check failed: " + name)
    path = SOURCE_DIR / name
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError("Snapshot was edited on disk. Restart with a clean runtime.")
    if not path.exists():
        path.write_bytes(data)
    loaded = sys.modules.get(path.stem)
    if loaded is not None and Path(getattr(loaded, "__file__", "")).resolve() != path.resolve():
        raise RuntimeError("Another rule version is loaded. Restart the runtime, then run all cells.")
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
importlib.invalidate_caches()
from rules import Game, action_id, action_move
from training import XiangqiEnv
from notebook_tools import ENDGAMES, action_mask, board_svg, replay_episode

OUTPUT_DIR = Path.cwd() / "xiangqi_output"
OUTPUT_DIR.mkdir(exist_ok=True)
try:
    from IPython.display import SVG, display
except ImportError:
    SVG = None  # A normal Python process can still run all rule checks.

def show_board(game, title="Xiangqi Lab"):
    svg = board_svg(game, title)
    path = OUTPUT_DIR / "board.svg"
    path.write_text(svg, encoding="utf-8")
    if SVG is not None:
        display(SVG(data=svg))
    else:
        print("Board saved:", path)

print("Rules ready. Snapshot:", SNAPSHOT_ID)
print("Project base commit:", SOURCE_BASE_COMMIT)
print("Rules: xiangqi-research-v1 | Episode format: xiangqi-lab.episode.v1")
'''
    code("load-source", bootstrap, cellView="form")
    md("checks-note", """
        ## 2. Check the environment

        Checks legal moves, rewards, game endings, and reset for the three endgames.
    """)
    code("checks", """
        def check_environment():
            for key, preset in ENDGAMES.items():
                probe = XiangqiEnv(max_plies=20)
                state, _ = probe.reset(fen=preset["fen"])
                assert probe.game.outcome() is None and state["legal_moves"]
                assert [len(state["board_planes"]), len(state["board_planes"][0]),
                        len(state["board_planes"][0][0])] == [14, 10, 9]
                assert sum(action_mask(probe)) == len(state["legal_moves"])
                for move, index in zip(state["legal_moves"], state["legal_action_ids"]):
                    assert action_move(index) == move and action_id(move) == index
                before = probe.game.export()
                try:
                    probe.step("a0a0")
                except ValueError:
                    pass
                else:
                    raise AssertionError("An illegal move was accepted")
                assert probe.game.export() == before
                next_state, reward, ended, limited, info = probe.step(state["legal_action_ids"][0])
                assert next_state["side_to_move"] != state["side_to_move"]
                assert info["acting_side"] == state["side_to_move"]
                print("PASS:", key, "— legal actions, move, and invalid move check")

            wins = [
                ("red", "4k4/3R1R3/9/9/9/9/9/9/9/3K5 w - - 0 1", "d8e8"),
                ("black", "3k5/9/9/9/9/9/9/9/3r1r3/4K4 b - - 0 1", "d1e1"),
            ]
            for winner, fen, move in wins:
                probe.reset(fen=fen)
                _, reward, ended, limited, info = probe.step(move)
                assert ended and not limited and reward == 1.0
                assert info["outcome"]["winner"] == winner
                assert info["rewards"][winner] == 1 and sum(info["rewards"].values()) == 0
                assert not any(action_mask(probe))
                print("PASS:", winner, "win — winner +1, loser -1")

            probe = XiangqiEnv(max_plies=2)
            probe.reset(fen=ENDGAMES["rook"]["fen"])
            for _ in range(2):
                _, reward, ended, limited, info = probe.step(probe.game.legal[0])
            assert limited and not ended and reward == 0 and not any(action_mask(probe))
            try:
                probe.step(probe.game.legal[0])
            except ValueError:
                pass
            else:
                raise AssertionError("An ended game accepted another move")
            probe.reset(fen=ENDGAMES["rook"]["fen"])
            assert probe.game.history == [] and probe.game.outcome() is None
            assert len(probe.game.repetitions) == 1 and sum(probe.game.repetitions.values()) == 1
            print("PASS: move limit, stop, and reset")
            print("All environment checks passed. No model was trained.")

        check_environment()
    """)
    md("presets-note", """
        ## 3. Choose a board

        Resets the chosen endgame and displays the board, FEN, and legal moves.
    """)
    code("choose-board", """
        PRESET_ID = "rook"  # @param ["rook", "cannon", "horse"]
        MAX_PLIES = 160     # @param {type:"integer"}
        SEED = 42          # @param {type:"integer"}

        if PRESET_ID not in ENDGAMES:
            raise ValueError("Choose rook, cannon or horse.")
        if type(MAX_PLIES) is not int or not 2 <= MAX_PLIES <= 1000:
            raise ValueError("MAX_PLIES must be an integer from 2 to 1000.")
        for key, item in ENDGAMES.items():
            board = Game(item["fen"])
            print(f"{key:7} | {item['name']:22} | {sum(bool(p) for row in board.board for p in row)} pieces")

        env = XiangqiEnv(max_plies=MAX_PLIES)
        obs, reset_info = env.reset(fen=ENDGAMES[PRESET_ID]["fen"], seed=SEED)
        rng = random.Random(SEED)  # The demo policy owns its random generator.
        show_board(env.game, ENDGAMES[PRESET_ID]["name"])
        print("Start FEN:", obs["fen"])
        print("Legal moves:", obs["legal_moves"])
    """)
    md("observation-note", """
        ## 4. View the state

        Shows the board data format, current player, and legal move IDs.
    """)
    code("inspect-state", """
        obs = env.game.observation()
        mask = action_mask(env)
        print("Board shape: (14, 10, 9)")
        print("Piece order:", obs["plane_order"])
        print("Turn:", obs["side_to_move"])
        print("Allowed actions:", sum(mask), "of", len(mask))
        print("Examples:", list(zip(obs["legal_moves"], obs["legal_action_ids"]))[:8])
        print("End state:", env.game.outcome())
    """)
    md("move-note", """
        ## 5. Make one move

        Makes one legal move and shows the updated board, reward, and game status.
    """)
    code("one-move", """
        MOVE = ""  # @param {type:"string"}
        if env.game.outcome() is not None:
            print("Game ended. Run 'Choose a board' to reset.")
        else:
            selected_move = MOVE.strip() or env.game.legal[0]
            obs, reward, terminated, truncated, info = env.step(selected_move)
            print("Move:", selected_move, "| Acting side:", info["acting_side"])
            print("Reward:", reward, "| Rewards by colour:", info["rewards"])
            print("Terminated:", terminated, "| Truncated:", truncated)
            print("Result:", info["outcome"])
            show_board(env.game, ENDGAMES[PRESET_ID]["name"])
    """)
    md("demo-note", """
        ## 6. Short automatic demo

        Plays up to `DEMO_PLIES` random legal moves and prints the move log.
    """)
    code("demo", """
        DEMO_PLIES = 6  # @param {type:"integer"}
        if type(DEMO_PLIES) is not int or not 0 <= DEMO_PLIES <= 20:
            raise ValueError("Use 0 to 20 plies for this short demo.")
        for _ in range(DEMO_PLIES):
            if env.game.outcome() is not None:
                break
            selected_move = rng.choice(env.game.legal)
            obs, reward, terminated, truncated, info = env.step(selected_move)
            env.game.history[-1]["actor"] = "colab-random-demo"
            print(f"{len(env.game.history):3} | {info['acting_side']:5} | {selected_move} | reward {reward:+.0f}")
            if terminated or truncated:
                break
        show_board(env.game, ENDGAMES[PRESET_ID]["name"])
        print("Result:", env.game.outcome() or "Game still in progress")
        print("No learning or weight update took place.")
    """)
    md("export-note", """
        ## 7. Save a game for the website

        Replays the game to check it, then saves a JSON file for the website.
        Downloads the file when `DOWNLOAD_JSON` is `True`.
    """)
    code("export", """
        DOWNLOAD_JSON = False  # @param {type:"boolean"}
        episode = env.game.export()
        episode["run_metadata"] = {
            "source": "colab-environment-demo", "trained": False,
            "preset_id": PRESET_ID, "seed": SEED,
            "source_base_commit": SOURCE_BASE_COMMIT, "source_sha256": SOURCE_HASHES,
        }
        replayed = replay_episode(episode)
        assert replayed.fen == env.game.fen
        assert replayed.outcome() == env.game.outcome()
        assert replayed.repetitions == env.game.repetitions
        # Keep previous exports when this cell is run again.
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        episode_path = OUTPUT_DIR / f"{PRESET_ID}-{stamp}.json"
        episode_path.write_text(json.dumps(episode, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Replay check passed.")
        print("Saved:", episode_path, "| Moves:", len(episode["moves"]))
        if DOWNLOAD_JSON:
            try:
                from google.colab import files
            except ImportError:
                print("Outside Colab: copy the saved file from the path above.")
            else:
                files.download(str(episode_path))
    """)
    md("import-note", """
        ## 8. Load a game

        Loads and checks a game from `IMPORT_PATH`, then displays its board.
        An empty path skips this cell.
    """)
    code("import", """
        IMPORT_PATH = ""  # @param {type:"string"}
        if IMPORT_PATH.strip():
            imported_episode = json.loads(Path(IMPORT_PATH).read_text(encoding="utf-8-sig"))
            imported_game = replay_episode(imported_episode)
            show_board(imported_game, "Game from the website")
            print("Moves:", len(imported_game.history), "| Result:", imported_game.outcome())
        else:
            print("Skipped. The environment setup is complete.")
    """)
    md("dqn-load-note", """
        ## 9. Load DQN

        Loads the DQN network, training loop, and model export tools.
    """)
    code("dqn-load", """
        import torch
        from dqn_training import TrainConfig, train, export_policy
        print("PyTorch:", torch.__version__)
        print("Training device:", "cuda" if torch.cuda.is_available() else "cpu")
    """, tags=["dqn"])
    md("dqn-config-note", """
        ## 10. Set training options

        Sets at least 1,000 training episodes and the validation rules for early stopping.
    """)
    code("dqn-config", """
        ALGORITHM = "double_dqn"  # @param ["dqn", "double_dqn"]
        TRAIN_PRESET = "rook"  # @param ["rook", "cannon", "horse"]
        MIN_EPISODES = 1000    # @param {type:"integer"}
        MAX_EPISODES = 3000    # @param {type:"integer"}
        EVAL_EVERY = 100       # @param {type:"integer"}
        EVAL_GAMES = 50        # @param {type:"integer"}
        PATIENCE = 5           # @param {type:"integer"}
        MIN_DELTA = 0.02       # @param {type:"number"}
        TRAIN_SEED = 42        # @param {type:"integer"}
        EPSILON_DECAY_STEPS = 100000  # @param {type:"integer"}

        if MIN_EPISODES < 1000:
            raise ValueError("Use at least 1,000 training episodes.")
        train_config = TrainConfig(
            preset=TRAIN_PRESET, learner="red", min_episodes=MIN_EPISODES,
            max_episodes=MAX_EPISODES, eval_every=EVAL_EVERY,
            eval_games=EVAL_GAMES, patience=PATIENCE,
            min_delta=MIN_DELTA, seed=TRAIN_SEED,
            epsilon_decay_steps=EPSILON_DECAY_STEPS,
            algorithm=ALGORITHM,
        )
        train_config.validate()
        print("Algorithm:", ALGORITHM)
        print("Train:", TRAIN_PRESET, "| Learner: Red | Opponent: random legal moves")
        print("Episode budget:", MIN_EPISODES, "to", MAX_EPISODES)
        print("Exploration: 1.0 to 0.1 over", EPSILON_DECAY_STEPS, "learner decisions.")
        print("After the minimum, stop after", PATIENCE, "validation checks without a clear gain.")
        print("Validation uses separate seeds on the same starting board.")
    """, tags=["dqn"])
    md("dqn-train-note", """
        ## 11. Train the model

        Starts training when `START_TRAINING` is checked, saves checkpoints, and stops on the validation rule.
        `RESUME_FOLDER` resumes an existing run.
    """)
    code("dqn-train", """
        START_TRAINING = False  # @param {type:"boolean"}
        RESUME_FOLDER = ""      # @param {type:"string"}

        training_result = None
        if START_TRAINING:
            from datetime import datetime, timezone
            if RESUME_FOLDER.strip():
                run_folder = Path(RESUME_FOLDER).expanduser().resolve()
            else:
                run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                run_folder = OUTPUT_DIR / "runs" / (train_config.algorithm + "-" + TRAIN_PRESET + "-" + run_name)
            training_result = train(
                train_config, run_folder, resume=bool(RESUME_FOLDER.strip()),
                source_hashes=SOURCE_HASHES,
            )
        else:
            print("Training is off. No training episodes have been run by this cell.")
    """, tags=["dqn"])
    md("dqn-export-note", """
        ## 12. Export the best model

        Packs the best checkpoint, policy code, and results into a ZIP for the website.
        Downloads it when `DOWNLOAD_MODEL` is checked.
    """)
    code("dqn-export", """
        SAVED_RUN_FOLDER = ""   # @param {type:"string"}
        DOWNLOAD_MODEL = False  # @param {type:"boolean"}

        saved_run = (Path(SAVED_RUN_FOLDER).expanduser().resolve() if SAVED_RUN_FOLDER.strip()
                     else training_result["run_dir"] if training_result else None)
        if saved_run is not None:
            model_zip = export_policy(saved_run)
            print("Model package:", model_zip)
            print("The website's model interpreter needs PyTorch. See the README inside the ZIP.")
            if DOWNLOAD_MODEL:
                try:
                    from google.colab import files
                except ImportError:
                    print("Outside Colab: copy the ZIP from the path above.")
                else:
                    files.download(str(model_zip))
        else:
            print("No training run selected. Train first or set SAVED_RUN_FOLDER.")
    """, tags=["dqn"])
    md("dqn-compare-note", """
        ## 13. Compare with the last DQN run

        Compares win rates and recent variation with the saved 100,000-step-exploration DQN run.
        Saves a table and chart after training finishes.
    """)
    code("dqn-compare", (ROOT / "notebooks/snippets/compare_dqn.py").read_text(encoding="utf-8"), tags=["dqn"])

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": TARGET.name, "provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "xiangqi_lab": {"stage": "dqn-ready-training-off", "source_base_commit": commit,
                            "source_sha256": hashes, "episode_schema": "xiangqi-lab.episode.v1"},
        }, "cells": cells,
    }


if __name__ == "__main__":
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(build_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(TARGET)
