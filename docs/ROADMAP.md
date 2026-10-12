# Research roadmap and implementation status

Status recorded on 2026-10-12. This separates the existing platform from the group's proposed research work. Method selection and experiments are still in progress; there are no trained-team-model results to report yet. Track task owners and progress in GitHub Issues rather than assigning names in this document.

## Implemented platform

| Area | Available now | Boundary |
| --- | --- | --- |
| Local website | Board editor, FEN/presets, matches, game export/import | In-memory local sessions; no hosted training service |
| Rules and environment | Legal actions, observations, one-ply `XiangqiEnv` | Simplified repetition rules; no Gymnasium spaces/subclass |
| Reference opponents | Random baseline and bundled Pikafish | Presets are not a curated, difficulty-labeled endgame dataset |
| Model integration | Persistent Python policy, JSON and UCI adapters, preflight | Team template is untrained; arbitrary weights are insufficient |
| Distribution | Windows and dual-architecture macOS package builder | Windows package tested locally; Mac launcher testing remains to be recorded |
| Collaboration | Source setup, handoff docs, PR/Issue templates, CI configuration | Actual workflow results are recorded by GitHub after upload |

## Proposed research sequence

The team's plan starts with endgames and fixed opponents, then considers learning from expert moves. The items below are planned, not implemented features.

| Stage | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 1. Define experiment contract | Curated endgame split and fixed-opponent wrapper | Tests for both learner colors, terminal states after either move, learner rewards and truncation |
| 2. Establish baselines | Reproducible random, weak and stronger fixed-opponent experiments | Saved configuration, seeds, opponent version/budget and held-out evaluation |
| 3. MCTS baseline | Search policy using the shared rules and legal actions | Correct value perspective across turns, stated search budget and comparison with random |
| 4. Expert move dataset | Collect positions and expert moves with labels/metadata | Versioned generator, provenance and train/evaluation separation |
| 5. Behavior cloning | Train a policy network on expert moves | Reproducible training command, checkpoint handoff and held-out results |
| 6. Optional PPO | Fine-tune policy/value networks if time and evidence justify it | Compatible learner environment, ablations and equal-budget evaluation |
| Later extension | Markov-game / jointly learning opponents | Explicit policy-update protocol and a suitable evaluation design |

A fixed-opponent wrapper should combine the learner's action with the opponent's response when play continues. It must also handle cases where either move ends the game, without calling a policy on a terminal position. Keep winner, draw and ply-limit truncation distinct. The underlying board game remains a fully observable, alternating, zero-sum game; fixing the opponent changes how the learner's transitions can be modeled.

## Suggested work boundaries

- **Platform:** browser interactions, model connection, episodes and portable packages.
- **Environment/evaluation:** fixed-opponent wrapper, endgame splits, reproducible match runner and metrics.
- **Search:** MCTS policy using the shared rules, with tests for alternating value perspective.
- **Learning:** expert-data generation, behavior cloning and any later PPO work; supply the [model handoff](MODEL_HANDOFF.md).

Choose owners in Issues. Agree on observation/action versions and reward semantics before parallel implementation so model and environment code remain compatible.

For every reported comparison, record the code commit, checkpoint hash, dataset split, seeds, opponent settings, search budget and hardware. Report wins/draws/losses, truncations, inference failures and time per move. Passing interface tests is an engineering result and should be reported separately from learning or playing-strength results.
