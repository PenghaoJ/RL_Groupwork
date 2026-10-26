# Research roadmap and implementation status

Current status, 26 October 2026: DQN/Double DQN training and evaluations have completed, teammate PPO models are integrated, and MCTS is pending. See [the current results](EXPERIMENT_STATUS_20261026.md) and [MCTS handoff](MCTS_HANDOFF.md). The tables below preserve the planning snapshot from 21 October 2026; statements about unstarted training or future PPO integration are historical, not the present status.

## Historical platform snapshot (21 October)

| Area | Available now | Boundary |
| --- | --- | --- |
| Local website | Board editor, FEN/presets, matches, game export/import | In-memory local sessions; no hosted training service |
| Rules and environment | Legal actions, observations, one-ply `XiangqiEnv` | Simplified repetition rules; no Gymnasium spaces/subclass |
| DQN training | Fixed-opponent wrapper, replay, target network, validation, early stopping, resume and model export | Formal training is off; same-preset validation does not establish unseen-board generalisation |
| Reference opponents | Random baseline and bundled Pikafish | Presets are not a curated, difficulty-labeled endgame dataset |
| Model integration | Persistent Python policy, JSON and UCI adapters, preflight | Team template is untrained; arbitrary weights are insufficient |
| Distribution | Windows and dual-architecture macOS package builder | Windows package tested locally; Mac launcher testing remains to be recorded |
| Collaboration | Source setup, handoff docs, PR/Issue templates, CI configuration | Actual workflow results are recorded by GitHub after upload |

## Original proposed research sequence (21 October)

The next work is to run and compare DQN, PPO and MCTS on an agreed endgame split and evaluation protocol. Expert-data and behavior-cloning work remains optional. The table below records follow-up experiments; the fixed-opponent wrapper itself is implemented.

| Stage | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 1. Define experiment contract | Curated endgame split and common opponent/evaluation settings | Shared positions, seeds, reward semantics and time limits |
| DQN experiment | At least 1,000 training episodes, validation early stopping and exported checkpoint | Saved metrics and held-out evaluation; engineering smoke tests alone are insufficient |
| 2. Establish baselines | Reproducible random, weak and stronger fixed-opponent experiments | Saved configuration, seeds, opponent version/budget and held-out evaluation |
| 3. MCTS baseline | Search policy using the shared rules and legal actions | Correct value perspective across turns, stated search budget and comparison with random |
| 4. Expert move dataset | Collect positions and expert moves with labels/metadata | Versioned generator, provenance and train/evaluation separation |
| 5. Behavior cloning | Train a policy network on expert moves | Reproducible training command, checkpoint handoff and held-out results |
| 6. Optional PPO | Fine-tune policy/value networks if time and evidence justify it | Compatible learner environment, ablations and equal-budget evaluation |
| Later extension | Markov-game / jointly learning opponents | Explicit policy-update protocol and a suitable evaluation design |

The implemented fixed-opponent wrapper combines the learner's action with the opponent's response when play continues and handles endings after either move. It keeps wins, draws and ply-limit truncations distinct. The underlying board game remains a fully observable, alternating, zero-sum game; fixing the opponent changes how the learner's transitions can be modeled.

## Suggested work boundaries

- **Platform:** browser interactions, model connection, episodes and portable packages.
- **Environment/evaluation:** fixed-opponent wrapper, endgame splits, reproducible match runner and metrics.
- **Search:** MCTS policy using the shared rules, with tests for alternating value perspective.
- **Learning:** expert-data generation, behavior cloning and any later PPO work; supply the [model handoff](MODEL_HANDOFF.md).

Choose owners in Issues. Agree on observation/action versions and reward semantics before parallel implementation so model and environment code remain compatible.

For every reported comparison, record the code commit, checkpoint hash, dataset split, seeds, opponent settings, search budget and hardware. Report wins/draws/losses, truncations, inference failures and time per move. Passing interface tests is an engineering result and should be reported separately from learning or playing-strength results.
