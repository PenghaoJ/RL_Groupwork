# Model handoff between teammates

The precise observation, action and process protocol is in [MODEL_API.md](../xiangqi_lab/MODEL_API.md). This guide describes what to commit and how another teammate can reproduce your setup. A checkpoint alone is not a working integration.

## Prepare a model folder

Copy [team_template](../xiangqi_lab/models/team_template/) into `xiangqi_lab/models/<your_policy>/`. Change its ID to a unique `custom-...` value and implement `Policy`. Keep the original template unchanged so CI can exercise a working baseline.

```text
xiangqi_lab/models/your_policy/
├── model.json          Portable manifest, including relative checkpoint filename
├── policy.py           Network construction, checkpoint loading and inference
├── requirements.txt    Only this model's dependencies, with tested versions
├── README.md           Setup, checkpoint download and evaluation instructions
└── weights.pt          Downloaded locally; ignored by Git
```

The extra files shown here are a handoff convention, not an automatic installer. List required Python version, supported OS/architectures and devices in the model README. Keep `"python": "{python}"` in shared Python manifests when the app and policy use the same environment, and use a relative checkpoint path such as `weights.pt`.

Windows teammate setup, after cloning and creating `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r xiangqi_lab/models/your_policy/requirements.txt
.\.venv\Scripts\python.exe -X utf8 xiangqi_lab/validate_model.py --model custom-your-policy
.\.venv\Scripts\python.exe -X utf8 xiangqi_lab/app.py
```

Replace `your_policy` and the ID with the real ones, and download the checkpoint to the documented location before validation. On macOS use `.venv/bin/python` and make the bundled engine executable as shown in the root README. If a framework needs a different Python version or platform-specific installation command, document and test that model environment explicitly; do not assume every ML package supports the platform's default Python.

For a separate interpreter, a manifest can use its executable path. Keep machine-specific copies in an ignored folder such as `models/local_alice_policy/` with their own unique ID, rather than committing a personal `C:/Users/...` or `/Users/...` path into the shared manifest.

## Share a checkpoint reproducibly

Share large weights as a separate team artifact or a GitHub Release attachment. Record a durable download location, exact filename, SHA-256 hash and the code commit needed to load it. A normal source clone does not fetch ignored weights.

In the model README, record:

- Algorithm, network architecture, input plane order, action encoding and legal-action masking.
- Python/library versions, CPU fallback if implemented, and devices actually tested.
- Checkpoint filename, download location, SHA-256, training code commit and configuration.
- Training endgame split, random seeds and opponent version/search settings.
- Evaluation split and results, or an explicit statement that evaluation is still pending.
- Any maximum search budget or known unsupported positions.

Calculate the checkpoint hash on Windows with `Get-FileHash <path> -Algorithm SHA256`, or on macOS with `shasum -a 256 <path>`. Do not replace missing measurements with expected results.

## Integration acceptance

1. Start from a fresh source checkout and follow only the model's documented setup.
2. Run `validate_model.py --model <id>`. Its three cases include Red and Black; passing verifies those legal outputs, not playing strength.
3. Load the model in the website with **Models → Refresh list → Test**. Choose it for Red or Black, click **Play**, and try both sides. Use **Save game** to export a replayable episode.
4. Evaluate against fixed opponents on held-out endgames with identical budgets. Report wins, draws, losses and truncations separately, plus inference time and any failures. Keep bulk output in ignored `output/` or `runs/`; commit a concise reproducibility summary.

Policies load once into a persistent process and may serve multiple browser sessions. Load weights in `__init__`, use the supplied observation each turn, and avoid hidden per-game state. An illegal move, timeout or inference error must be reported as a failure, not silently replaced with a random move in an evaluation result.

## Training semantics to agree on first

The base environment advances one ply and returns the acting side's reward. `dqn_training.FixedOpponentEnv` now provides learner moves plus fixed-opponent replies, opponent openings, terminal checks after either move, learner-perspective rewards and time-limit handling. The DQN notebook exports a complete model folder ZIP; its interpreter requires PyTorch. See the [Colab guide](COLAB.md) for training, validation limits and checkpoint handoff. Do not plug the base environment's alternating rewards directly into a single-agent trainer.
