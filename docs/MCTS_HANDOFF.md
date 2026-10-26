# MCTS teammate handoff

Read `EXPERIMENT_STATUS_20261026.md` first. Reuse the existing rules and website interface; implement your search or learned search as a new policy. This package does not contain a finished MCTS algorithm or MCTS checkpoint.

## Choose the task

1. **Pure MCTS baseline:** search at each move using the rules and rollouts. Report simulations, rollout depth, seed and actual time per move. This need not have trained weights or a training-episode count.
2. **Learned MCTS:** train a policy/value network from self-play or other explicitly documented data, then use it during search. Keep the network, search code and training settings together. This is a separate training pipeline, not a rename of DQN.

Start with a legal pure-search baseline, then decide whether the course requires a learned network. There is no existing MCTS training run to resume in this handoff. Reference: [DeepMind's AlphaZero overview](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) and [OpenSpiel AlphaZero documentation](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md).

## Files to read

| File | Purpose |
| --- | --- |
| `xiangqi_lab/rules.py` | `Game`, legal moves, endings, move/action conversion |
| `xiangqi_lab/training.py` | One-player-ply environment and terminal rewards |
| `xiangqi_lab/MODEL_API.md` | Exact observation and policy interface |
| `xiangqi_lab/models/team_template/` | Runnable interface example, not MCTS |
| `xiangqi_lab/board_bank.py` | Split checks and balanced training-start sampling |
| `experiments/rook20_20261025/boards.json` | Twenty training and six validation starts |
| `xiangqi_lab/policy_perspective.py` | Optional Red/Black view conversion |
| `docs/MODEL_HANDOFF.md` | Model delivery and acceptance checks |
| `notebooks/MCTS_Handoff_Start.ipynb` | Environment setup and one cloned simulation; no trainer |

Use `rules.py` and `training.py` in cloud Python directly. Do not make the cloud notebook call a teammate's `127.0.0.1` website: that address means the cloud machine itself. The local browser is for visualization and inference, not a requirement for simulations.

## State, rewards and search

- Fully observable, alternating, deterministic, zero-sum game under the project's research rules. With a fixed opponent, the learner wrapper can be treated as an MDP.
- One base `step` is one player's move, not a full Red/Black turn. Default game cap: 160 plies (80 full turns).
- `XiangqiEnv.step` returns reward from the **acting player's** view. `info['rewards']` gives both players' terminal rewards. Winner +1, loser -1, draw 0; ordinary moves and captures have no shaping reward in the experiments. `Game.step` itself returns an outcome, not a numeric reward.
- No legal move is a loss, including stalemate. Threefold position-and-side repetition and 120 noncapture plies are draws. A 160-ply limit is a separate truncation. Report limits separately and document any value target used at a search/episode cutoff.
- Use legal moves only. The action space is 8100; IDs are `source_square * 90 + destination_square`. The API describes board coordinates and the 14 planes.
- **FEN or board planes alone do not include the repetition history.** Preserve the full `repetition_counts`, capture clock, side and current ply/cap. Reconstructing `Game(fen)` alone during a match loses history and resets progress toward the cap.
- When the full `Game` is available, `copy.deepcopy(game)` preserves its search state. When implementing the website's observation-to-search-state bridge, test it against the original `Game` after nonzero plies, captures and repeated positions; preserve elapsed plies and counters before searching. Do not mutate the supplied observation or real game during simulations.
- Choose one value convention for MCTS backups. If values are stored from the player-to-move perspective, convert signs when backing across opposing turns; if stored from the root-player perspective, handle the opponent's minimization consistently. Test with immediate wins/losses for both colors. Do not let both players maximize the same player's reward.
- Each Python policy worker is reused for multiple games. Keep no hidden game history unless it is explicitly keyed and validated; rebuilding search from the supplied state is a safe first version.

## Data boundaries

`experiments/rook20_20261025/boards.json` has separate `training` and `validation` arrays. Train or generate learning trajectories only from the 20 training starts. Use the six validation starts for model/search-setting selection. If using the existing schedule, visit every training start once per shuffled block of 20 episodes. Record actual training steps and runtime, not just episode counts.

`output/dqn/UNSEEN10_FOUR_20261025/boards.json` is **evaluation only**. Do not add these ten starts or their mirrors to training, replay, rollout-generated training datasets or tuning. It is normal to search from a test state at evaluation time, but freeze learned parameters and the search settings and do not retain its trajectories as training data. These boards have already informed project discussion, so reserve another untouched bank before claiming a new blind test. Only starting-board separation is guaranteed; all reachable state sets are not proven disjoint.

Do not tune on the final match results. If you train against the supplied frozen opponents, report that new opponent distribution explicitly; it differs from the earlier random-opponent training and cannot be described as the same controlled configuration.

## Website adapter

Copy `models/team_template` to `models/mcts_teammate`, use unique ID `custom-mcts-teammate`, and keep `python` set to `{python}`. Use `checkpoint: null` for pure search, or a relative weight filename for learned search. Start the server using the environment with your dependencies.

```python
class Policy:
    def __init__(self, checkpoint=None, device="cpu"):
        # Load search settings and optional trained weights once.
        pass

    def select_move(self, observation, time_budget_ms=800):
        # Search a faithful copy of this state and return one legal move.
        raise NotImplementedError("Implement the MCTS search here.")
```

Return a UCI-style coordinate move such as `a3a4`, or `{'action_id': integer}`. The above is a contract sketch, not a completed search policy. Set and respect a search deadline; `time_budget_ms` is advisory for Python policies, and the manifest's timeout is a hard worker limit. An 800-ms MCTS setting is not automatically equal compute to one neural forward pass. Log actual latency and search nodes/simulations. Record failures rather than silently replacing failed moves with random ones.

Run `python xiangqi_lab/validate_model.py --model custom-mcts-teammate`. Then use **Models → Refresh list → Test**, select both players and press **Play**. Passing the three-position preflight means legal sample outputs, not proven strength or complete rule correctness.

## Suggested work order

1. Run the starter notebook or local environment smoke check. Explain the files and current experiment status back to the teammate.
2. Implement the state bridge and basic MCTS. Test legal moves, side/value signs, history-sensitive endings, cloning and the time cap.
3. Validate the website adapter on both sides; run short matches against random. Do not reuse Double DQN's weights as an MCTS checkpoint.
4. Tune only on the validation split. If building learned MCTS, save named best and last checkpoints plus optimizer/RNG state, settings and logs. Use a separate run directory. Agree a budget and suitable local/cloud runtime before a long paid run.
5. Freeze the policy. Evaluate 200 games on the ten-board/20-opponent-seed bank against random to match the existing protocol, then pair each starting FEN with both player assignments against frozen Double DQN and PPO. Preserve seeds for randomized search. Deterministic repeated games on the same board are not independent evidence.
6. Return a complete model folder: `model.json`, `policy.py`, all imported helper files, relative weights if any, search settings, `requirements.txt`, README, hashes, test results and replayable episodes. For pure MCTS, return code/settings instead of invented weights.

## Provided opponents and limits

The handoff includes the original selected Double DQN and teammate PPO seed 42, with unchanged weight hashes and portable copied manifests. They can serve as frozen opponents after installing their dependencies. Both were trained as Red; Black uses the verified view transform. They have not been trained on both roles. PPO seed 43/44 and the other old/new weights remain with the project owner, not in this compact handoff.

This is a **source-and-model handoff**, not an offline Python installer. Rules and the basic web app use the standard library; the neural opponents require PyTorch and PPO packages. Windows was checked locally. macOS has a bundled Pikafish binary but this exact neural dependency environment has not been tested on macOS; use CPU and a supported Python 3.12 environment. Linux cloud can run the rules and neural policies, but no Linux Pikafish binary is bundled. Do not claim all platform/GPU combinations were tested.
