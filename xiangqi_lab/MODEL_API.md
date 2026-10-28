# Model integration contract (v1)

The website is an inference and evaluation host. It cannot infer a network architecture from arbitrary `.pt`, `.pth` or `.onnx` weights. A complete drop-in model is a folder containing **configuration, inference code and any required weights**, with a working Python environment. The supplied template is runnable but untrained.

The selected Double DQN, PPO and MCTS v4 are already bundled. Use `start.py` or the root launchers for their dependency setup; see [shared models](../docs/SHARED_MODELS.md). Other custom models still need their own compatible dependencies.

The untrained JSON example (`custom-example`) and team template (`custom-team-template`) are hidden from the website's player menus and Models list. Their code, manifests and CLI validation remain available. When copying a template, assign a new unique ID as described below so your model appears in the website.

Observations also include `initial_fen`, the unchanged starting FEN of the current game. Use this field for preset-specific routing: a persistent worker may serve different games, and Black's first request is usually at ply 1. Existing board/action/history fields keep their original meaning.

## Recommended: drop-in Python policy

1. Copy `models/team_template` to `models/my_policy`.
2. Change the unique `id` and display `name` in `model.json`.
3. Put your weights in that folder; set `checkpoint` to their relative filename.
4. Implement `Policy` in `policy.py`. Load the correct architecture and weights in `__init__`. Return a legal move from `select_move`.
5. If using PyTorch, ONNX Runtime or other dependencies, set `python` to your inference environment's executable. The bundled Python contains only the standard library. Install dependencies in your own environment, not the web app's runtime.
6. Open **Models → Refresh list → Test**. After the check passes, choose the model for Red or Black and click **Play**.

```json
{
  "schema": "xiangqi-lab.model.v1",
  "id": "custom-my-policy",
  "name": "Our endgame policy v1",
  "description": "Checkpoint from experiment 12",
  "protocol": "python",
  "entrypoint": "policy.py",
  "python": "C:/our-env/python.exe",
  "checkpoint": "weights.pt",
  "device": "cpu",
  "timeout_seconds": 30
}
```

On macOS, use the environment's path such as `/Users/you/project/.venv/bin/python`. `{python}` selects the interpreter running the server (the bundled interpreter when using a portable launcher, or your development environment when running from source). For a shared manifest, prefer `"python": "{python}"` and start the server with the environment containing your model dependencies; this avoids committing a teammate's personal interpreter path. `device` is passed to your class; the host does not enable a GPU or silently change devices. Set `checkpoint` to `null` for policies without weights. `entrypoint` must stay inside the model folder. Inference runs from that folder, so use the supplied absolute checkpoint path when loading weights.

```python
class Policy:
    def __init__(self, checkpoint=None, device="cpu"):
        # Load your architecture and checkpoint ONCE here.
        # Set evaluation mode; disable training-time randomness as appropriate.
        self.model = load_your_model(checkpoint, device)

    def select_move(self, observation, time_budget_ms=800):
        legal = observation["legal_moves"]
        # Your implementation must mask illegal actions.
        move = your_inference(self.model, observation, legal)
        return move
```

`load_your_model` and `your_inference` are placeholders for the team's implementation, not bundled functions. For a working standard-library example, use `models/team_template/policy.py`.

Accepted returns:

```python
return "a3a4"
return {"move": "a3a4"}
return 4905
return {"action_id": 4905}
```

Return exactly one representation. Boolean IDs, out-of-range IDs, malformed responses and illegal moves are rejected. A policy process is loaded once, reused across moves and serialized per model. Do not keep hidden per-game state in the policy: the same worker may serve several browser sessions. Use the provided observation, or define a separate explicit stateful protocol in a later extension. Clicking **Refresh list** in **Models** terminates cached workers so edited code and weights are loaded on the next request.

Import-time and inference `print()` calls are redirected to stderr by the Python worker. Exceptions and model timeouts stop the requested move without changing the board. A failed worker restarts on its next request. `timeout_seconds` covers startup plus inference on the first call, and inference on later calls; valid range is 0.1–120 seconds. `time_budget_ms` is advisory for Python/JSON policies, and is enforced as UCI `go movetime` for UCI engines. Windows/macOS environment paths are platform-specific: adjust them on the teammate's machine.

## Observation and action representation

Every request contains `protocol: "xiangqi-lab.agent.v1"`, `time_budget_ms`, and `observation`:

| Field | Contract |
| --- | --- |
| `fen` | Current position; `w` means Red, `b` means Black. |
| `board_planes` | 14 × 10 × 9 binary array. Row 0 is Black's home rank, row 9 is Red's. Column 0 is file a. |
| `plane_order` | `KAEHRCPkaehrcp`: king, advisor, elephant, horse, rook, cannon, pawn, first Red then Black. |
| `side_to_move` | `red` or `black`. The planes are never flipped for Black. |
| `legal_moves` | Allowed UCI coordinate moves, e.g. `a3a4`. |
| `legal_action_ids` | IDs corresponding to `legal_moves`, in the same order. |
| `halfmove_clock` | Plies since the last capture. |
| `ply`, `max_plies` | Experiment progress and truncation limit. |
| `repetition_counts` | Position-FEN-and-side keys mapped to occurrence counts; used by the research draw rule. |

Action space size is 8100. `id = (src_row * 9 + src_col) * 90 + dst_row * 9 + dst_col`. For example `a3a4` is `54*90+45 = 4905`. Use `rules.action_id` / `rules.action_move` to convert. For policy logits, mask all IDs not in `legal_action_ids` before choosing an action. The planes alone omit counters needed for a Markov state under this environment's draw/truncation rules.

Observation formats, IDs and saved episodes do not depend on the UI language. Stable IDs, not translated display names, select models.

## External JSON and UCI agents

Existing `models.json` registrations remain supported. You may also place a `model.json` under a model folder with the same `schema` and a `json` or `uci` protocol.

```json
{
  "schema": "xiangqi-lab.model.v1",
  "id": "custom-external",
  "name": "External policy",
  "protocol": "json",
  "command": ["{python}", "{model_dir}/agent.py"],
  "timeout_seconds": 30
}
```

`{root}` is the app directory, `{model_dir}` the manifest directory, `{python}` the running server interpreter. `command` is an argument array, never shell code. JSON agents start once per move, read one JSON line from stdin and output one JSON response (`move` or `action_id`) before exiting. Diagnostics go to stderr. Large neural models should prefer the persistent Python policy interface.

UCI agents use `"protocol": "uci"` and a command pointing to the executable. The host sends `uci`, `isready`, `ucinewgame`, `position fen ... moves ...`, `go movetime ...` and expects `bestmove ...`. Startup handshakes time out after 10 seconds; search has a 12-second overhead allowance. Custom engines must be built for the destination machine. The bundled Pikafish handles platform CPU dispatch; the app cannot make an arbitrary incompatible executable portable.

## Preflight from the command line

From the app directory, using the Python that starts the app:

```text
python validate_model.py --model custom-my-policy
python validate_model.py --all
```

For a Windows release, replace `python` with `runtime\python\python.exe`. This is also available from **Models → Test** in the UI. It calls the real adapter on three independent positions, including both Red and Black, and validates legal outputs without changing the current match. Passing demonstrates those sample calls, not model strength or correctness on every possible endgame. Validate your own checkpoint's full evaluation suite separately.

## Local HTTP interfaces

All POST bodies are JSON. Only the local origin is accepted; the app is not a public API server.

| Endpoint | Input / purpose |
| --- | --- |
| `GET /api/meta` | Presets, discovered model IDs, available languages and engine presence. |
| `POST /api/models/reload` | `{}`; reread manifests and stop cached Python workers. |
| `POST /api/models/check` | `{"model":"custom-my-policy"}`; isolated three-position preflight. |
| `POST /api/session` | `{}`; create a game session. Returns `session` and `state`. |
| `POST /api/observation` | `session`, `revision`; read the model observation and outcome. |
| `POST /api/reset` | `session`, `revision`, `fen`, `max_plies`. |
| `POST /api/move` | `session`, `revision`, `move`; apply a legal move. |
| `POST /api/engine` | `session`, `revision`, `model`, `milliseconds`; infer, validate, then apply a model move. |
| `POST /api/export` | `session`, `revision`; export a replayable episode. |

Always use the latest returned `revision` for session requests. Stale writes return HTTP 409 and the current state. Bad input/illegal moves return 400; model runtime failures/timeouts return 500. An unsuccessful inference does not increment the revision or alter the position. There is no browser file-upload endpoint for executing models; teammates add trusted code to the local models folder.

## Training

`training.XiangqiEnv` remains available without a browser. `reset(fen=...)` returns `(obs, info)`. `step(move_or_action_id)` advances one side's ply and returns `(obs, reward, terminated, truncated, info)`. Reward is from the acting side's perspective; use `info.rewards` for both sides' terminal targets. It is a lightweight API, not a Gymnasium subclass. See README.md for simplified repetition rules and the distinction between fixed-opponent MDPs and two-player learning.
