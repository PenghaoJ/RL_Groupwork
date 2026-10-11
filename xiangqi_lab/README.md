# Xiangqi Lab — Endgame Workspace v1.1.1

An offline research workspace for setting up Xiangqi endgames, comparing policies and exporting reproducible episodes. The browser communicates with a local Python server; Pikafish runs through UCI. No account or internet connection is required for the bundled features.

## Start

Extract the complete release ZIP before running it.

| Package | Launcher | Included runtime |
| --- | --- | --- |
| Windows x64 | `Start-Windows.bat` | Python 3.13.16, universal Pikafish, NNUE |
| macOS Intel / Apple Silicon | `Start-macOS.command` | Both Python architectures, universal Pikafish, NNUE |

The launcher opens your default browser. The default address is `http://127.0.0.1:8765`; occupied ports are skipped. Keep the launcher window open and use Ctrl+C to stop. Python, Node, Docker and pip installation are not required for the website, baseline policies or bundled engine. Move the whole extracted folder together.

The Windows package has been exercised on this Windows x64 computer. The Mac package preserves executable permissions and symbolic links but has not been run on a Mac. macOS may require confirming the downloaded, unnotarized application in Privacy & Security. If an extractor loses launcher permissions, run `bash Start-macOS.command` from Terminal. Unsupported/old operating systems and Windows ARM are not covered by these packages.

## Language

A fresh browser profile defaults to English. Use the header's Language selector to choose Chinese; the preference is saved locally. Switching language preserves the board, selected controllers, editing draft and history. Both interface languages retain traditional Xiangqi characters on the pieces, player icons and editing palette, as well as the traditional river inscription. Menus, accessible piece names and tooltips follow the selected interface language.

All application-owned Chinese UI strings and traditional board symbols are in **`static/locales/zh-CN.json`**. To distribute an application without Chinese characters, remove that file and reload the page: the server stops advertising Chinese, the language selector disappears, the board falls back to K/A/E/H/R/C/P labels and RIVER, and a previously saved Chinese preference falls back to English. Core code, English HTML, API messages, protocol fields and model templates do not depend on that file. Third-party licenses and source archives are preserved as supplied.

## Play and create positions

- Select a preset or enter FEN. Presets are illustrative positions, not labeled solved puzzles.
- Build a position by choosing a piece and placing it on the board. Use the eraser to remove pieces. Apply validates both kings, reachable squares and the side to move.
- Choose a controller for Red and Black independently. Start match runs model turns and waits for human turns. Model step performs one inference.
- Pause prevents subsequent moves; the current inference is allowed to finish. Undo removes one ply. Reset restores the episode's initial position.
- Ply limits apply when loading or resetting. Hitting the limit truncates an episode rather than declaring a draw.
- Export an episode to save initial FEN, moves, controllers, before/after positions, search information and outcome. Import replays and validates every move. Imported search information is not recomputed.

Each browser page has an independent in-memory session. Refreshing a page starts a new session. Export your episode before refreshing or shutting down. The service listens only on loopback and is not an internet-facing or multi-user deployment.

## Hand this to the model team

Start with **[MODEL_API.md](MODEL_API.md)** and the working **`models/team_template/`** folder.

1. Copy the folder, add weights, set a unique ID/name in `model.json`.
2. Load your network in `Policy.__init__` and implement `select_move` in `policy.py`.
3. Set the manifest's `python` to an environment with your model's dependencies, if required.
4. Open Connect a model, Reload models, and Check integration.
5. Assign the model to Red or Black and start a match.

Python policies load once into a persistent process. JSON command-line and UCI adapters remain available. Integration checks use three sample positions covering both sides and do not change your game. They establish that those calls return legal moves, not that arbitrary weights will work or that a policy is strong. The app cannot reconstruct a network architecture from a weights file alone.

The bundled Python has no PyTorch, NumPy or ONNX Runtime. Teams must supply the appropriate inference environment for their model. Device and dependency compatibility remain the model team's responsibility. Failed inference, illegal moves and timeouts leave the board unchanged and report an error.

The random baseline, JSON example and team template are all **untrained**. This release contains no training loop, MCTS trainer, GPU management or trained team checkpoint.

## Training environment

`training.py` provides a standard-library-only `XiangqiEnv`. It uses Gymnasium-like return conventions but is not a Gymnasium subclass and does not declare spaces.

```python
from training import XiangqiEnv
from rules import PRESETS

env = XiangqiEnv(max_plies=80)
obs, info = env.reset(fen=PRESETS[0]['fen'])
obs, reward, terminated, truncated, info = env.step(obs['legal_moves'][0])
```

One step is one player's ply. Planes have shape 14×10×9 in `KAEHRCPkaehrcp` order, always with Black's home rank at row 0. Side to move is explicit. Actions use UCI strings or integer IDs in `[0, 8099]`; the API guide defines the mapping and legal-action mask.

Reward is from the acting player's perspective: terminal win +1, loss -1, otherwise 0. `info.rewards` reports both players' terminal targets, so the losing side must also receive its terminal target when constructing training returns. The environment transition is deterministic; the optional reset seed does not seed external policies/searchers.

Observations include counters and repetition counts, not only piece planes. This is a fully observable turn-based zero-sum game. For a fixed, memoryless opponent, combine your move and its response into a single-agent transition if you want an MDP wrapper. If both policies change during learning, use a Markov-game or multi-agent formulation. Full observability alone does not establish the Markov property.

## Rule scope

Standard moves, horse legs, elephant eyes, cannon screens, palace/river restrictions, self-check and facing kings are implemented. Both checkmate and stalemate lose. The research draw policy is threefold position repetition including side to move, or 120 non-capture plies. This does **not** implement competition perpetual-check/chase responsibility. Pikafish's internal search adjudication can differ from these simplified research rules. Keep the environment rules fixed when comparing experiments and document the limitation.

## Files and checks

| File | Purpose |
| --- | --- |
| `app.py` | Local HTTP server, session isolation, model-check endpoints |
| `rules.py`, `training.py` | Legal rules, observations, episodes and training environment |
| `engines.py` | UCI and model inference bridge |
| `model_registry.py` | Drop-in manifests and legacy model registry |
| `model_worker.py`, `policy_process.py` | Persistent Python inference processes |
| `models/team_template/` | Working drop-in policy example |
| `MODEL_API.md` | Model contract and local HTTP interfaces |
| `validate_model.py` | Browser-free model integration checks |
| `static/` | Offline web UI and optional Chinese locale |
| `engine/`, `third_party/` | Binaries, weights, license notices and corresponding engine source |
| `tests/` | Rules, adapter, isolation, timeout and integration tests |

From the app folder, run `python -m unittest discover -s tests -v` or `python validate_model.py --all`. In a Windows release use `runtime\python\python.exe` in place of `python`. In the development workspace use `../.runtime/python/python.exe`. The workspace's `tools/build_release.py` builds both platform ZIPs and SHA256 checksums.

## Third-party components

Pikafish 2026-09-06 comes from its [official release](https://github.com/official-pikafish/Pikafish/releases/tag/Pikafish-2026-09-06). Engine GPLv3 terms, author notices and the corresponding source archive are included. NNUE has separate terms, including a restriction on unapproved commercial use; see `engine/NNUE-License.md`.

Windows Python is the official Python.org 3.13.16 full ZIP. macOS Python uses Astral python-build-standalone 20261003. Runtime license files are preserved. URLs and provenance are listed in `third_party/SOURCES.txt`.
