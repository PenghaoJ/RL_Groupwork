# Development guide

Start with the repository [README](../README.md). All commands below run from the repository root unless stated otherwise. Python 3.13 is the development and CI baseline; the app and tests have no third-party Python dependencies.

## Runtime and architecture

```text
Browser (static/index.html, app.js, style.css)
    -> local JSON HTTP requests
app.py (sessions, revisions, request validation)
    -> rules.py (positions, legal moves, outcomes, episodes)
    -> engines.py (choose and validate a controller's move)
        -> Pikafish subprocess through UCI
        -> Python policy through policy_process.py + model_worker.py
        -> external JSON / UCI process

Offline training -> training.XiangqiEnv -> the same rules.py
```

| Change | Main location | What to check |
| --- | --- | --- |
| Board, controls, layout | `xiangqi_lab/static/` | Actual browser flow; both interface languages |
| HTTP endpoints and sessions | `xiangqi_lab/app.py` | HTTP tests; stale revision and session isolation |
| Rules, observations and action encoding | `xiangqi_lab/rules.py` | Rules tests and Pikafish legal-move comparison |
| Training transition semantics | `xiangqi_lab/training.py` | Rewards, termination, truncation and acting side |
| Model discovery | `xiangqi_lab/model_registry.py` | Unique IDs, manifest paths and validation errors |
| Inference and worker lifetime | `engines.py`, `policy_process.py`, `model_worker.py` inside `xiangqi_lab/` | Legal moves, worker reuse, exceptions and timeout tests |
| One team's policy | `xiangqi_lab/models/<name>/` | Model preflight and the team's evaluation suite |
| Packaging | `tools/build_release.py` | Full extraction and launch on each target OS |

The app is a local inference and game host. Training should call the Python environment directly, without driving the web UI. It has no training-job scheduler, automatic ML dependency installer or GPU manager.

## Daily commands

Windows PowerShell, after creating `.venv` as described in the README:

```powershell
.\.venv\Scripts\python.exe -X utf8 xiangqi_lab/app.py --port 8765 --no-browser
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s xiangqi_lab/tests -v
.\.venv\Scripts\python.exe -X utf8 xiangqi_lab/validate_model.py --model custom-team-template
```

Stop the server before proceeding to the other commands in that terminal, or open a second terminal. On macOS, replace the executable with `.venv/bin/python`; first run `chmod +x xiangqi_lab/engine/Pikafish-MacOS-universal`. Using the environment's executable directly avoids shell-activation issues.

Add `--all` instead of `--model custom-team-template` to validate every installed controller. This also executes locally added models and requires their dependencies and checkpoints. The CI workflow deliberately runs the shared suite and the supplied template; it does not install or evaluate every teammate's neural model.

For a model with extra dependencies, install its documented requirements in your own environment and start the server with that environment's Python. A manifest with `"python": "{python}"` then uses the same interpreter. Core app dependencies remain separate from optional training/model dependencies.

## Automated and manual checks

[tests.yml](../.github/workflows/tests.yml) runs on pushes to `master`, pull requests, and manual dispatch. It uses Python 3.13 on Windows x64, macOS Intel and macOS Apple Silicon, with the checked-in engine. The Actions tab shows actual results after the workflow is pushed and GitHub Actions is enabled. Workflow jobs have read-only repository access and do not deploy or upload artifacts.

When updating CI, check the official [checkout action](https://github.com/actions/checkout), [setup-python action](https://github.com/actions/setup-python) and [runner labels](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

The suite covers legal moves, action conversion, draw/truncation behavior, isolated HTTP sessions, episode replay, policy-process reuse, malformed responses, timeouts and real engine integration. Do not remove engine tests merely to make a platform green. There is no Linux engine bundled in the repository.

For UI changes, manually load a position, make a legal move, run a model turn, undo and save/load a game. Check English and Chinese menus, traditional piece symbols, and the English-only fallback when testing a copy without the optional locale. Model code changes can be picked up with **Models → Refresh list**; backend code needs a restart. Reloading the page creates a new game session.

These source-level checks do not test OS download prompts, double-click launchers or the extracted offline runtime. Record actual platform checks separately when preparing a release.

## Portable packages

Development from source uses an installed Python; offline releases carry their own Python. The existing builder requires these local inputs, which are intentionally absent from Git:

| Input | Location |
| --- | --- |
| Complete Windows Python runtime | `.runtime/python/`, including `python.exe` and its libraries |
| Apple Silicon Python standalone archive | `.setup/mac-runtimes/*-aarch64-apple-darwin-install_only_stripped.tar.gz` |
| Intel Mac Python standalone archive | `.setup/mac-runtimes/*-x86_64-apple-darwin-install_only_stripped.tar.gz` |

Exact runtime provenance is listed in [SOURCES.txt](../xiangqi_lab/third_party/SOURCES.txt). Keep only the intended archive for each Mac architecture in that input folder. Ensure the version constants in the builder match the inputs. Then run `python tools/build_release.py` with Python 3.13. It builds **both** OS ZIPs and `dist/SHA256SUMS.txt`.

The current builder assumes those inputs are prepared and copies the app tree, including locally added model files. Build from a clean release copy containing only intended models and assets, with both platforms' runtimes prepared. It does not install model dependencies or upload releases. Preserve the runtime/engine notices, extract the outputs into a new location, and verify their launchers on Windows and both Mac architectures before describing a release as tested there.

## Troubleshooting

| Symptom | Next check |
| --- | --- |
| Portable launcher says to extract the complete ZIP | A source checkout has no bundled runtime. Use the README's source commands. |
| `py` / `python3` not found | Install Python 3.13, reopen the terminal and check its version. |
| Mac engine reports permission denied | Apply the `chmod +x` command above. OS download approval is separate from file permissions. |
| Opening `index.html` directly does not work | Start `app.py` and open the HTTP address printed in the terminal. |
| Browser shows an older copy of the app | Check the printed port; an existing server may have made the new one select another port. |
| `No module named torch` or another ML library | Check the model's Python environment and dependency instructions. The portable runtime is stdlib-only. |
| Model not listed or duplicate ID error | Check `model.json`, unique `custom-...` ID and entrypoint; use **Models → Refresh list**. |
| Model fails its first move | Run `validate_model.py --model <id>`; inspect checkpoint path, shape, device and legal-action masking. |
