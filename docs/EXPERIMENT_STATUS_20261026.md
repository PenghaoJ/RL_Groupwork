# Project status — 26 October 2026

This is the current handoff summary. Older notebook comments and planning documents may describe an earlier stage. Local files include work not yet committed or pushed. A GitHub clone alone is not this handoff snapshot.

## Built and tested

- Local Python server with an HTML/CSS/JavaScript board, English menus by default, optional Chinese menus, and traditional piece symbols.
- Presets, a board editor, FEN import, human/model and model/model play, legal moves, game endings, and JSON game export/import.
- A framework-free two-player environment (`rules.py`, `training.py`), plus a separate fixed-opponent learner wrapper.
- DQN and Double DQN training, legal-action masks, validation, early stopping, checkpoint export and resume support. Colab notebooks include the rules; the local website is not needed for training.
- Twenty fixed rook training starts and six validation starts. Ten separate test starts (four rook, three cannon, three horse) have been used repeatedly for evaluation, never as training starts. They are now a regression set, not a new blind holdout.
- Three teammate PPO checkpoints imported. Their 600 recorded games and the selected Double DQN's 200 recorded games were reproduced exactly with unchanged weights.
- Both selected policies were trained as Red. A tested board rotation and color swap allows Black inference. This changes the view, not the trained weights or the defensive experience.

## Results against the random legal opponent

Each row below uses 200 cases from the same ten-board/opponent-seed bank, with no learning during evaluation. Draws and move-limit truncations remain separate.

| Checkpoint | Wins | Losses | Draws | Limits |
| --- | ---: | ---: | ---: | ---: |
| Original DQN | 23 | 0 | 170 | 7 |
| Original Double DQN (selected) | 33 | 0 | 158 | 9 |
| ROOK20 DQN | 25 | 1 | 168 | 6 |
| ROOK20 Double DQN | 18 | 0 | 167 | 15 |
| Teammate PPO seed 42 | 84 | 0 | 108 | 8 |
| Teammate PPO seed 43 | 54 | 1 | 137 | 8 |
| Teammate PPO seed 44 | 58 | 0 | 142 | 0 |

The later terminal-experience sampling experiment reached 1,500 episodes for each algorithm, selecting episode 300 for each. DQN won 19/200 and Double DQN 14/200 on the same regression test. It did not beat the selected original Double DQN. Original weights were preserved. These are checkpoint comparisons, not matched-compute claims about all algorithms.

## Direct match

The selected old Double DQN (best episode 200) played PPO seed 42 (best episode 700) on each of the ten test starts, swapping player assignments on the same FEN: 20 deterministic games total.

- Double DQN: 0 wins; PPO: 2 wins; 18 repetition draws; no move-limit games.
- PPO's two wins were as Red on rook positions.
- A separate familiar website rook preset was won by Red Double DQN in 27 plies. It is not part of the 20-game total.
- PPO seed 42 was the preview choice and had the highest delivered test score of the three seeds. This is exploratory evidence, not fresh blind model selection.

## Received MCTS v4 match

The teammate's v2, v3 and v4 folders are successive experience snapshots with identical policy code, each containing isolated rook, cannon and horse memories. The user selected v4 before testing. No training or changes to either agent were made.

| Set | Games | Double DQN wins | MCTS v4 wins | Draws | Limits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing ten test starts, both assignments | 20 | 1 | 13 | 4 | 2 |
| Three familiar presets, both assignments | 6 | 1 | 4 | 0 | 1 |

Custom starts use the received search-only path. Familiar presets explicitly select their matching scene memory for either player assignment. MCTS searches natively as either side, but its stored experience contains only Red actions. The original Double DQN retains its existing Black perspective adapter.

MCTS used its delivered inference defaults: at most 96 simulations, depth-10 rollouts and an 800 ms requested move budget. The game cap was 160 plies. Fixed seeds were paired by board; wall-clock stopping prevents exact seed-only reproducibility. This is a small comparison of the delivered agents, not an equal-compute algorithm ranking.

A reproducible received-code defect remains unchanged: a forced one-ply winning rollout returns -1 instead of +1 from its starting player's perspective. Its impact on these win counts has not been measured. All 26 completed games were replayed successfully, and checkpoint, source and experience hashes stayed unchanged. A checker correction and discarded partial game are documented in `resume_audit.json`; the first 20 completed games were preserved.

Results, charts and an offline replay viewer are in `output/matches/DoubleDQN_vs_MCTS_v4_20261026/`. Received source snapshots are in `output/mcts/RECEIVED_20261026/`. The updated Part E draft, `reports/PartE_Results_DoubleDQN_PPO_MCTS_20261026.docx`, includes two MCTS charts, match settings and limitations. The earlier PPO report is preserved. Word pagination remains for the user's review under the agreed draft workflow.

## Diagnosis and limits

Recorded ROOK20 games contain positions where a one-move win was available but the policy chose a repetition draw. Checks found that reward and repetition handling followed the implemented rules. Sparse decisive rewards, limited experience per starting board and learned value errors are plausible contributors; their individual causal effects were not established. Changing terminal sampling alone did not fix performance.

Research rules use threefold position-and-side repetition and 120 plies without a capture. They do not implement full tournament perpetual-check/chase adjudication. No claim is made that these hand-created starts represent all Xiangqi endgames or are all forced wins.

The earlier MCTS handoff ZIP did not contain an implemented MCTS agent or trainer. A teammate's MCTS v4 has since been archived and tested locally as described above. Update on 28 October: the selected original Double DQN, PPO seed 42 and MCTS v4 are now bundled as [website players](SHARED_MODELS.md), with a stateless initial-position routing adapter for MCTS. The earlier handoff ZIP is unchanged. Its PPO inference loader and frozen weights are included, but a complete PPO training/resume workspace is not. Those opponent checkpoints are for inference, not exact training resumption: optimizer, replay buffer and training RNG state are absent.

## Evidence in the handoff ZIP

- `output/dqn/UNSEEN10_FOUR_20261025/boards.json`: ten evaluation-only starts and opponent seeds.
- `handoff/evidence/four_models_comparison.json`: original versus ROOK20 results.
- `handoff/evidence/terminal50_comparison.json` and `terminal50_verification.json`: final terminal-sampling evaluation and verification.
- `handoff/evidence/ppo_verification.json`: 800 reproduced games and color-view checks.
- `output/matches/DoubleDQN_vs_PPO42_20261026/`: 20 paired match replays, a separate website demo, and report.
- `SNAPSHOT.json` and `SHA256SUMS.txt`: source provenance and file integrity. Not all historical weights or raw training outputs are included.
