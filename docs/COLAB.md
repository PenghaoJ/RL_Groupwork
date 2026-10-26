# Colab: board rules and DQN

## Twenty-board experiment

Use [Xiangqi_Rook20_DQN_DoubleDQN_20261025.ipynb](../notebooks/Xiangqi_Rook20_DQN_DoubleDQN_20261025.ipynb) for the separate multi-board experiment. It contains the rules, trainer and [fixed board bank](../experiments/rook20_20261025/boards.json), so it does not need GitHub or the website at runtime.

Both algorithms start from the same seed-42 random initialization. Each block of 20 training episodes visits the 20 training starts once in a shuffled order. Six other starts are used only for validation, with 10 fixed opponent seeds each. Every 100 episodes, validation chooses the best checkpoint and updates early stopping. Training runs for at least 1,000 and at most 3,000 episodes; five checks without an improvement above 0.02 stop the run after the minimum. The two runs can therefore finish at different episode counts. Actual decisions and board plies are logged.

The ten boards from the October 25 test are excluded, including their horizontal mirrors. They are not used for checkpoint selection. All new rook starts place the red pawn on rank 4 or higher, which also prevents reaching the four old rook test starts whose pawn is on rank 3. The validation split separates starting boards; it does not claim that all reachable states are disjoint.

Open the new notebook with a fresh runtime, run sections 1 and 2, check `START_TRAINING`, then run section 3. It trains DQN and then Double DQN under distinct names: `DQN_ROOK20_20261025_S42` and `DoubleDQN_ROOK20_20261025_S42`. The source repository copy keeps training off. `RESUME` restores an unfinished run only with the same board-bank hash, source snapshot and training settings. Completed runs are loaded rather than trained again.

Each new best model gets a named export and a backup link. Keep those files outside the temporary Colab runtime. `last.pt` contains the optimizer and replay state for resuming; the website model ZIP contains only inference weights and reports. The earlier two best models remain in `output/dqn/FAIR_COMPARISON_20261022/models/` and have extra backups under `output/dqn/ROOK20_20261025_S42/baselines/`. Output and weight folders are ignored by Git.

Rebuild the new notebook after editing its sources with `python tools/build_rook20_notebook.py`. The six source files are `rules.py`, `training.py`, `notebook_tools.py`, `dqn_model.py`, `dqn_training.py` and `board_bank.py`. `python -m unittest discover -s xiangqi_lab/tests -p test_board_bank.py -v` checks split exclusion, legal generation, balanced sampling, frozen validation, training updates and exact resume behavior for both algorithms.

## Original fixed-board notebook

Open [the team Colab notebook](https://colab.research.google.com/drive/10YsasfU1bvfekR505h7D3QopAXVjsolC), or upload [the repository copy](../notebooks/DQN_ChineseChess.ipynb) to Colab. Restart the runtime after updating the source snapshot, then choose **Runtime → Run all**. `START_TRAINING` is **False** in the generated repository copy, so this runs the environment demo and loads the training settings without starting formal training. The live Drive copy may have training enabled for an active experiment; check it before running all cells.

Sections 1–8 use the same `rules.py` and `training.py` as the website, plus `notebook_tools.py` for display and replay, with no ML dependency. Sections 9–12 add `dqn_model.py` and `dqn_training.py` and require PyTorch 2.6 or later (below 3); Colab normally provides it. For a separate local environment, install `torch>=2.6,<3` from the official PyTorch package source. The local smoke tests used Python 3.12.14 and PyTorch 2.14.1+cpu; no GPU test is claimed.

The notebook does not start the website, connect to localhost, install Pikafish, fetch GitHub, or need the Windows/macOS offline package. Its included source snapshot makes it usable before repository changes have been pushed.

## Use it

1. Run all cells. The environment checks should print `All environment checks passed` and a board should appear.
2. In **Choose a board**, select `rook`, `cannon` or `horse`. Run that cell and the cells below it to reset and switch boards. No random starting boards are generated.
3. **Make one move** accepts a legal move such as `c4c5`, or uses the first legal move if left blank. **Short automatic demo** uses random legal moves for both sides, with no learning. Set its length to zero to skip.
4. In **Save a game**, set `DOWNLOAD_JSON = True` and run the cell. Open the local website, select **Load game / 载入对局**, and choose the downloaded JSON. It restores the initial position, move history and final board, not model weights.
5. To go the other way, save a game in the website, upload the JSON through Colab's Files sidebar, and set `IMPORT_PATH` in the optional import cell. It validates and displays a separate board.

Colab's temporary files can be lost when the runtime resets. Download the JSON you need to keep. The notebook is saved on Drive; its runtime output files are separate.

## Interface and limits

- `XiangqiEnv.reset(fen=..., seed=...)` returns `(observation, info)`.
- `XiangqiEnv.step(move_or_action_id)` returns `(observation, reward, terminated, truncated, info)` and advances **one player's ply**. A full Red-and-Black turn takes two calls.
- Observations include 14 × 10 × 9 board planes, side to move, legal moves and IDs, halfmove clock, ply count, ply limit and repetition counts. Board planes alone omit state needed for the draw rules.
- Action IDs range from 0 to 8099, with the same coordinates and mapping as the website. `action_mask(env)` returns 8100 booleans and all-false after an episode ends. The base observation can still list geometrically legal moves after a draw or truncation, so check the outcome rather than relying on a non-empty move list.
- The scalar reward is for the **acting side**. `info['rewards']` reports both colours: winner +1, loser -1; nonterminal play, draws and ply-limit truncation return 0. There is no capture bonus. Illegal moves raise `ValueError` before changing the game, with no penalty transition.
- Stop on `terminated or truncated`, then call `reset` explicitly. The caller will later choose whether to repeat a preset or use a separate test set.
- Rules use threefold position repetition and 120 non-capture plies for draws. Checkmate and stalemate are losses. Full tournament perpetual-check/chase adjudication is not implemented.
- `reset(seed=...)` only records the seed. The demo explicitly creates `random.Random(SEED)` for reproducible move choices.

## DQN training and early stopping

The defaults are:

| Setting | Default |
| --- | --- |
| Algorithm | `double_dqn` in the current notebook experiment; `dqn` remains available |
| Starting position | `rook`, the existing fixed rook-and-pawn endgame |
| Learner / opponent | Red DQN / fixed uniform-random legal policy |
| Minimum / maximum training episodes | 1,000 / 3,000 |
| Episode limit | 160 original board plies |
| Validation | 50 games every 100 training episodes |
| Early-stop patience | 5 validation checks after the minimum budget |
| Minimum clear improvement | More than 0.02 in validation mean reward |
| Exploration decay | 100,000 learner decisions in the current notebook experiment |
| Start automatically | No |

Training episodes include wins, losses, rules-based draws and time-limit truncations. Validation games and the random-policy baseline do not count toward the minimum training budget. After the minimum is reached, early stopping tracks validation mean reward `(wins - losses) / games`. Patience starts at episode 1,000; with the default schedule and no further clear improvement, the earliest automatic stop is episode 1,500. The 3,000-episode cap provides a separate upper bound. Interruptions and runtime resets can still stop before the minimum; they are not reported as a completed 1,000-episode run.

Every strictly better validation score saves `best.pt`, even before episode 1,000. Early-stop patience uses the separate `min_delta` threshold. The best checkpoint may therefore come from an earlier episode than the final one. The final training result restores the best weights for use and export.

To start later on a suitable runtime, run sections 1–10, review the settings, then check `START_TRAINING` in section 11 and run that cell. Training writes each completed episode to `episodes.csv`, validation results to `validation.csv`, and prints progress every 25 episodes. It saves `best.pt`, `last.pt`, a replayable example game, configuration and a summary in a new timestamped run folder. No board is rendered during the training loop. Section 12 creates a ZIP containing the best weights, Python policy adapter, requirements and reports for the website. The model interpreter needs PyTorch; the portable website's bundled Python does not include it. See [model handoff](MODEL_HANDOFF.md).

`RESUME_FOLDER` resumes from that folder's latest `last.pt`. Keep the same source snapshot and training settings; only the maximum episode budget and device may change. A resume restores the optimizer, replay memory, target network, early-stop state and random generators. Keep both `last.pt` and `best.pt` and the run's CSV files. A checkpoint is written after validation; work after the last checkpoint may need to be repeated. Download the run folder or copy it to persistent storage before disconnecting. The model ZIP is for inference and does not contain the replay buffer or resumable training checkpoint.

### Training semantics

`FixedOpponentEnv` in `dqn_training.py` combines one learner action with the opponent reply if the game continues. It checks endings after either move and returns reward from the learner's colour. An optional opponent opening is handled when the learner is Black. The ply limit is aligned downward to a learner-turn boundary, by at most one ply, so the next state at a time limit is suitable for learner-value bootstrapping.

The DQN uses learned piece embeddings, a small network for the current board, a pooled representation of **every entry** in the repetition-count map, and an 8,100-action output head. The capture clock and side to move are included. The full repetition map is retained in replay; a current-board-only encoding would omit relevant draw history. The externally imposed move cap is excluded from the value-network input. This is a function-approximation baseline, not a guarantee that the network learns an optimal state representation.

Training uses experience replay, a target network, Huber loss, clipped gradients and epsilon-greedy exploration. Epsilon decreases from 1.0 to 0.1 over `EPSILON_DECAY_STEPS` learner decisions. The original `TrainConfig` default is 20,000; the current notebook explicitly selects 100,000. Other defaults are learning rate 0.0003, discount 0.99, replay capacity 10,000, batch size 64, 512 warmup decisions, one update per four learner decisions and target-network copying every 250 updates. A learner decision usually covers two original environment plies; both counts are logged. `TrainConfig.algorithm` defaults to `dqn` for existing callers; the notebook now explicitly selects `double_dqn`.

### Longer-exploration experiment

The original 20,000-decision run completed 1,500 episodes and 63,541 learner decisions, stopping after five checks without a clear gain. Its best checkpoint was at episode 100: 5 wins, 0 losses and 45 draws in 50 validation games (score 0.10). These are same-preset validation results, not unseen-board test results.

The longer-exploration run changed only `EPSILON_DECAY_STEPS` to 100,000. It completed 1,500 episodes and 82,473 learner decisions. Its best checkpoint, at episode 100, won all 50 validation games; the final checkpoint won 4/50, with 46 draws. The early peak was not sustained. Both runs used seed 42, the same preset, opponent, network, reward, optimizer, validation bank and stopping settings. These validation results selected checkpoints; they are not independent test results.

### Double DQN comparison

The next experiment changes only `ALGORITHM` from `dqn` to `double_dqn`, keeping the 100,000-decision exploration schedule and all other training settings. Leave `RESUME_FOLDER` empty for the same seeded initialization. Training writes into a new algorithm-prefixed timestamped folder. The earlier DQN model and logs remain separate.

Standard DQN uses the target network to select and value the best legal next action. Double DQN selects that action with the online network, then reads its value from the target network. Both paths mask illegal actions; terminal values stay zero and time-limit transitions retain bootstrapping. Network architecture and website inference are unchanged. See the [Double DQN paper](https://arxiv.org/abs/1509.06461).

Exports use distinct names and website IDs: `dqn_rook_model.zip` / `custom-dqn-rook`, and `double_dqn_rook_model.zip` / `custom-double-dqn-rook`. Both contain the best validation checkpoint. Existing DQN checkpoints remain usable for inference; switching algorithms during resume is rejected. Automatic downloads remain off.

Compare best and final validation win rates, the mean and range of the final five validation checks, and the drop from the best to final checkpoint. Report actual learner decisions and run lengths, since early stopping can produce different totals. A single training seed is exploratory evidence; repeat promising settings before claiming a reliable gain. Separately test saved best models with a fresh opponent-seed bank before claiming generalisation.

Section 13 uses the saved validation log from `rook-20261021T135047973987Z` as the DQN reference. It checks the other training settings, then saves `comparison.json` and, when matplotlib is available, `comparison.png` in the new run folder. It reports the final five checks' mean and standard deviation together; low variation at a low win rate is not an improvement. The reference is specific to this experiment. Change it before comparing another board or training setup. The code comes from `notebooks/snippets/compare_dqn.py`.

Both action selection and the target-network maximum are restricted to legal actions. Checkmate, stalemate and rules-based draws stop bootstrapping. A time-limit truncation ends the episode but retains its next-state legal actions for bootstrapping. The environment-demo helper's all-false mask after truncation is not used to construct DQN targets. These choices follow the [DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning) and [termination/truncation distinction](https://farama.org/Gymnasium-Terminated-Truncated-Step-API).

### What early stopping can establish

Validation uses a fixed bank of opponent seeds separate from the training seeds. It uses greedy learner actions, no parameter updates, and separate environments and random generators. Wins, losses, draws and truncations are reported separately. This measures performance on the **same starting board** against the same opponent policy. It can detect a plateau or decline there; it does not prove resistance to overfitting to the preset, opponent or validation seed bank. No new random boards are generated. Later evaluation must reserve unseen endgames and a separate test seed bank; `evaluate(..., split="test")` supports a separate seed namespace, but unseen test boards still need to be supplied in a future extension. Do not repeatedly use test results to choose checkpoints.

The [Colab FAQ](https://research.google.com/colaboratory/faq.html) lists chess training among restrictions for free managed runtimes without a positive compute-unit balance; it does not distinguish Xiangqi. Check applicable runtime terms before starting formal training. The notebook leaves training off by default.

## Keep the notebook in sync

After changing any of the five source files included in the snapshot, run from the repository root:

```text
python tools/build_colab_notebook.py
python -m unittest discover -s xiangqi_lab/tests -p test_colab.py -v
python -m unittest discover -s xiangqi_lab/tests -p test_dqn.py -v
```

The DQN tests need PyTorch and skip in the standard-library-only app environment. They exercise only short test episodes, including real parameter updates, checkpoint resume, legal-action targets, the 1,000-episode early-stop guard and exported-model calls through the actual website worker. They do not establish playing strength or replace a full training run.

Commit the updated sources and notebook together. Upload the generated notebook to Colab; GitHub pushes do not automatically update the existing Drive notebook. Restart the Colab runtime after changing source snapshots.

SHA-256 hashes in the notebook identify the included source exactly, with newlines normalised to LF for Windows/macOS/Linux consistency. `source_base_commit` records the repository HEAD at build time; the file hashes, not that commit alone, identify any included uncommitted source. Outputs and execution counts are cleared in the generated notebook. Do not edit its embedded source by hand: edit the project source and rebuild it.
