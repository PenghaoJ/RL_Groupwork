# Play with the three shared agents

The repository includes the selected Double DQN, PPO and MCTS v4 payloads. A normal `git pull` downloads them; no separate model ZIP, training run or personal Python path is needed.

## Start

1. Pull the latest `master` and close your old Xiangqi server.
2. Windows: double-click **Start-Xiangqi-Windows.bat**. macOS: open **Start-Xiangqi-macOS.command**.
3. On the first run, keep the terminal open while missing CPU model packages are installed into `xiangqi_lab/.venv-models`. Internet access and free disk space are needed. Later runs reuse the environment. Nothing is installed into the system Python.
4. Select a board. In the Red and Black player lists, select **Double DQN**, **PPO** or **MCTS v4**. Press **AI move** or **Play**. Choose **Human** to play yourself.

If the Mac launcher is not executable, run `chmod +x Start-Xiangqi-macOS.command` from the repository folder. macOS download/security restrictions may require opening it through Finder's Open command.

A source checkout does not include Python itself. Windows accepts Python 3.10–3.13, or can reuse Python from the complete Windows release ZIP placed beside the launcher. For a Mac source checkout, install Python 3.12 from [python.org](https://www.python.org/downloads/macos/) if no supported runtime is present. Apple Silicon can also use Python 3.13. Intel Mac needs Python 3.10–3.12 and the older CPU dependency set selected by the launcher: [PyTorch stopped Intel Mac builds after 2.2](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).

Manual start with a compatible Python: `python xiangqi_lab/start.py`. Use `--check-models` to check dependencies without installing or opening the site, or `--setup-models` to install dependencies without starting the server. Starting `app.py` directly bypasses setup; its interpreter must already contain the model packages.

The bootstrap installs CPU PyTorch plus NumPy, Gymnasium and SB3 Contrib. MCTS alone uses only the standard library. An install failure is shown in the terminal and stops startup; it never silently substitutes a random player. On Windows, startup logs are under `%LOCALAPPDATA%/XiangqiLab/logs`.

## Included versions

| Menu name | Shared folder | Selected payload |
| --- | --- | --- |
| Double DQN | `xiangqi_lab/models/double_dqn` | Original best, episode 200 |
| PPO | `xiangqi_lab/models/ppo` | Teammate rook20 seed 42 best, episode 700 |
| MCTS v4 | `xiangqi_lab/models/mcts_v4` | Received v4 experience and original search code |

The first two were trained as Red. Their existing adapters rotate and swap colors when playing Black; this enables play but does not prove equal strength on both sides. They can produce legal moves on other positions, which is not evidence of reliable strength across every endgame or the full game. PPO retains its original capacity of 162 distinct repetition positions; keep the website's default 160-ply limit for this checkpoint.

MCTS uses the initial preset to select the rook/cannon/horse memory bank. The full-board preset merges the four received banks; custom starting positions use search only. The website supplies `initial_fen` on every request so Black's first turn and interleaved browser games route correctly. Its search limits remain 96 iterations and rollout depth 10, subject to think time; it does not update the saved experience while playing.

The received MCTS rollout has a known terminal-value sign issue on odd-length rollouts. The integration deliberately preserves that original algorithm and experience, matching the version investigated in the existing comparison. The small adapter fixes website scene routing only. New website games are not a rerun of the earlier benchmark.

Payload SHA-256 values are recorded in `xiangqi_lab/models/shared_provenance.json`; original local models, experiments and results are preserved. Only these selected small payloads are included in ordinary Git. Other checkpoints and personal AI-use declarations remain ignored.

## Verification

Run `python -m unittest discover -s xiangqi_lab/tests -p test_shared_models.py -v` with the model environment. It checks hashes, portable manifests, MCTS scene changes and legal inference on preset boards as both colors. Neural checks explicitly skip if the optional packages are absent. These are integration checks, not new training or strength measurements. Windows is checked locally; the workflow also exercises the selected dependency sets on macOS, whose results must be checked in GitHub Actions.
