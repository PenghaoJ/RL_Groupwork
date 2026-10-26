# RL Groupwork — Xiangqi Endgame Lab

A local Xiangqi workspace for our endgame learning project. Set up positions, play against Pikafish or a custom policy, and export games for analysis. The interface defaults to English, with optional Chinese menus and traditional Xiangqi piece symbols in both languages.

**Current stage (26 October 2026):** the platform and model interfaces work; DQN and Double DQN experiments have completed. Three teammate PPO checkpoints were imported and their 600 recorded test games reproduced. The selected old Double DQN and PPO seed 42 also played a 20-game paired match. MCTS is the next teammate's implementation task. Training remains off by default in source notebooks. Weights and detailed results are ignored by Git; a source clone alone does not include them. See [current results](docs/EXPERIMENT_STATUS_20261026.md) and the [MCTS handoff](docs/MCTS_HANDOFF.md).

## Start here

| Your task | Guide |
| --- | --- |
| Run the complete offline ZIP | [Application guide](xiangqi_lab/README.md) |
| Start on Windows with one double-click | Keep `Start-Xiangqi-Windows.bat` beside the full Windows release ZIP and open the BAT |
| Run or change the GitHub source | Quick start below, then [development guide](docs/DEVELOPMENT.md) |
| Submit a change with your teammates | [Contributing](CONTRIBUTING.md) |
| Integrate a trained policy | [Model handoff](docs/MODEL_HANDOFF.md) and [API contract](xiangqi_lab/MODEL_API.md) |
| Use the rules and DQN trainer in Colab | [Colab guide](docs/COLAB.md) and [notebook](notebooks/DQN_ChineseChess.ipynb) |
| Choose the next research task | [Roadmap and current status](docs/ROADMAP.md) |

## Source quick start

Install **Python 3.13** and Git, then clone the repository, or use GitHub Desktop's clone feature:

```text
git clone https://github.com/PenghaoJ/RL_Groupwork.git
cd RL_Groupwork
```

The web app and its baseline tests use only Python's standard library. No npm build, pip packages or GPU are needed. The repository includes the Windows x64 and macOS universal Pikafish engines and their network file.

On **Windows**, run from the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -X utf8 xiangqi_lab/app.py
```

On **macOS**, use a Python 3.13 installation and run from the repository root:

```bash
python3 --version
python3 -m venv .venv
chmod +x xiangqi_lab/engine/Pikafish-MacOS-universal
.venv/bin/python -X utf8 xiangqi_lab/app.py
```

The server opens a browser, usually at `http://127.0.0.1:8765/`. Read the terminal for the actual port. Keep that terminal open; press Ctrl+C to stop. Python source changes require a server restart; refresh the browser after frontend edits.

**A GitHub clone or “Download ZIP” is a source checkout.** It does not contain the ignored Python runtimes or prebuilt release ZIPs. On Windows, double-click `Start-Xiangqi-Windows.bat`: it checks the local app, a bundled runtime, a local `.venv` and installed Python 3.13. If Python is missing, place the complete Windows release ZIP beside it; the launcher unpacks that package and uses its Python with your source code. It can also ask you to select the ZIP. It does not download or install Python. For manual source development, use the commands above. The Mac launcher still requires the extracted macOS offline package; use the source commands on Mac for a Git checkout. GitHub Pages cannot run this Python backend or its local engine.

For teammates who only want to run the site, share **both** `dist/Start-Xiangqi-Windows.bat` and `dist/XiangqiLab-Windows-v1.2.0.zip`. Keep them together and double-click the BAT; manual extraction is optional. The launcher is self-contained, so it can also run when a ZIP viewer extracts only the BAT to a temporary folder. It searches the launcher's folder, parent, their `dist` folders, Downloads and Desktop (no recursive disk scan); if the package is elsewhere, a file picker asks for it. A launcher alone cannot recreate missing application files or Python.

Automatic extraction and timestamped startup logs go to `%LOCALAPPDATA%\XiangqiLab\`. The console prints the actual app folder; add models to that folder's `models/` directory. Repeated launches reuse the package; a different ZIP is kept separately, and incomplete existing copies are preserved. Keep the console open while playing. Windows security or organisation policy may still require approving a downloaded BAT or block PowerShell; the launcher does not change system policy or request administrator access.

## Repository layout

```text
RL_Groupwork/
├── xiangqi_lab/              Current web application, rules and model interfaces
│   ├── static/              HTML, CSS, JavaScript and optional Chinese locale
│   ├── models/team_template/ Runnable, untrained policy example
│   ├── tests/               Rules, HTTP and model integration tests
│   ├── engine/              Bundled Pikafish binaries, NNUE and notices
│   └── third_party/         Engine source archive and component provenance
├── docs/                    Development, model handoff and research roadmap
├── notebooks/               Colab rules demo and optional DQN training
├── .github/                 Automated checks and issue / pull request templates
├── tools/build_release.py   Offline package builder (requires local runtimes)
├── reports/                 Coursework progress documents
└── ChineseChessBoard_Package/ Original desktop GUI, retained for reference
```

New web-platform work belongs in `xiangqi_lab/`. The root launchers `Start-Xiangqi-Windows.bat` and `启动残局实验室.bat` open the web app; `启动象棋.bat` opens the older desktop GUI. The web launchers and `xiangqi_lab/Start-Windows.bat` are generated from `tools/windows_launcher.ps1` by `tools/build_launchers.py`.

Local `.venv/`, `.runtime/`, `.setup/`, `dist/`, `output/`, training runs and model checkpoint files are excluded from Git. See [model handoff](docs/MODEL_HANDOFF.md) for sharing weights without putting them in ordinary commits.

## Check your changes

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s xiangqi_lab/tests -v
.\.venv\Scripts\python.exe -X utf8 xiangqi_lab/validate_model.py --model custom-team-template
```

On macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. The suite invokes the actual bundled engine. Linux has no bundled engine in this repository and is not a full-suite target.

[GitHub Actions configuration](.github/workflows/tests.yml) runs these checks on Windows x64, macOS Intel and macOS Apple Silicon. A configuration file is not evidence of a successful run: check the repository's Actions tab after pushing. Browser behavior and double-clicking a portable package still need manual checks on the target system.

## Research boundaries

The Python backend uses `http.server`; the frontend uses plain HTML/CSS/JavaScript and SVG. It binds to this computer's loopback address and keeps game sessions in memory. Export games before closing or refreshing the page.

`XiangqiEnv` advances **one player's ply** per step and returns that acting player's reward. The separate `FixedOpponentEnv` in `dqn_training.py` combines a learner move and opponent reply, with learner-perspective rewards. The optional DQN trainer requires PyTorch; the web application remains standard-library-only. See the [Colab guide](docs/COLAB.md) and [roadmap](docs/ROADMAP.md). Research draw rules differ from full competition adjudication; see the [application guide](xiangqi_lab/README.md).

Pikafish and its network have their own licenses and distribution notices in [engine/](xiangqi_lab/engine/) and [third_party/SOURCES.txt](xiangqi_lab/third_party/SOURCES.txt). Preserve these files when distributing the project. This repository does not currently declare a separate license for the team's original code.
