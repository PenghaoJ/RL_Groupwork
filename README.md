# RL Groupwork — Xiangqi Endgame Lab

A local Xiangqi workspace for our endgame learning project. Set up positions, play against Pikafish or a custom policy, and export games for analysis. The interface defaults to English, with optional Chinese menus and traditional Xiangqi piece symbols in both languages.

**Current stage:** the platform and model interfaces are implemented. Training methods and model selection are still being developed. The included random policy and team template are untrained; no trained team checkpoint, MCTS implementation, behavior-cloning trainer or PPO trainer is included.

## Start here

| Your task | Guide |
| --- | --- |
| Run the complete offline ZIP | [Application guide](xiangqi_lab/README.md) |
| Run or change the GitHub source | Quick start below, then [development guide](docs/DEVELOPMENT.md) |
| Submit a change with your teammates | [Contributing](CONTRIBUTING.md) |
| Integrate a trained policy | [Model handoff](docs/MODEL_HANDOFF.md) and [API contract](xiangqi_lab/MODEL_API.md) |
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

**A GitHub clone or “Download ZIP” is a source checkout.** It does not contain the ignored Python runtimes or prebuilt release ZIPs. The `Start-Windows.bat` and `Start-macOS.command` launchers expect the complete platform-specific offline package. For source development, use the commands above. GitHub Pages cannot run this Python backend or its local engine.

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
├── .github/                 Automated checks and issue / pull request templates
├── tools/build_release.py   Offline package builder (requires local runtimes)
├── reports/                 Coursework progress documents
└── ChineseChessBoard_Package/ Original desktop GUI, retained for reference
```

New web-platform work belongs in `xiangqi_lab/`. The root launcher `启动残局实验室.bat` opens the web app through its portable launcher; `启动象棋.bat` opens the older desktop GUI. Keep the original folder and launcher paths when sharing existing packages.

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

`XiangqiEnv` currently advances **one player's ply** per step and returns that acting player's reward. The planned fixed-opponent training wrapper, search and learning experiments are tracked in the [roadmap](docs/ROADMAP.md). The research draw rules are simplified and differ from full competition adjudication; see the [application guide](xiangqi_lab/README.md).

Pikafish and its network have their own licenses and distribution notices in [engine/](xiangqi_lab/engine/) and [third_party/SOURCES.txt](xiangqi_lab/third_party/SOURCES.txt). Preserve these files when distributing the project. This repository does not currently declare a separate license for the team's original code.
