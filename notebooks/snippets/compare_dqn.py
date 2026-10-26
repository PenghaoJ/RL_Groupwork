# Compare this run with the saved DQN run; do not update either model.
import csv
import json
from pathlib import Path
import statistics

reference_rates = [1.0, 0.04, 0.0, 0.02, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.08]
reference_episodes = list(range(100, 1501, 100))
reference_config = {
    "preset": "rook", "learner": "red", "min_episodes": 1000, "max_episodes": 3000,
    "max_plies": 160, "eval_every": 100, "eval_games": 50, "patience": 5,
    "min_delta": 0.02, "seed": 42, "learning_rate": 0.0003, "gamma": 0.99,
    "replay_size": 10000, "batch_size": 64, "warmup_steps": 512,
    "train_every": 4, "target_every": 250, "epsilon_start": 1.0,
    "epsilon_end": 0.1, "epsilon_decay_steps": 100000, "device": "auto",
}

if not training_result or training_result["config"].algorithm != "double_dqn":
    print("No Double DQN run to compare.")
else:
    comparison_folder = Path(training_result["run_dir"])
    current_config = json.loads((comparison_folder / "config.json").read_text())
    changed = {key: [value, current_config.get(key)] for key, value in reference_config.items()
               if current_config.get(key) != value}
    if changed:
        raise ValueError("Comparison settings differ: " + str(changed))
    with (comparison_folder / "validation.csv").open() as stream:
        comparison_rows = list(csv.DictReader(stream))
    if not comparison_rows:
        raise ValueError("No validation results yet.")
    current_rates = [float(row["win_rate"]) for row in comparison_rows]
    current_episodes = [int(row["episode"]) for row in comparison_rows]

    def rate_summary(rates, episodes, steps, best_episode):
        recent = rates[-5:]
        return {"best_checkpoint_win_rate": rates[episodes.index(best_episode)],
                "final_win_rate": rates[-1], "recent_mean_win_rate": statistics.mean(recent),
                "recent_std_win_rate": statistics.pstdev(recent), "recent_win_rates": recent,
                "recent_episodes": episodes[-5:], "training_episodes": episodes[-1],
                "learner_steps": steps, "best_episode": best_episode}

    comparison = {
        "reference_run": "rook-20261021T135047973987Z",
        "current_run": comparison_folder.name,
        "scope": "Same fixed rook board and random opponent; validation, not independent testing.",
        "dqn": rate_summary(reference_rates, reference_episodes, 82473, 100),
        "double_dqn": rate_summary(current_rates, current_episodes,
                                   training_result["summary"]["learner_steps"],
                                   training_result["summary"]["best_episode"]),
    }
    print("Metric                         DQN        Double DQN")
    for key, label in [("best_checkpoint_win_rate", "Best checkpoint win rate"),
                       ("final_win_rate", "Final win rate"),
                       ("recent_mean_win_rate", "Last 5 checks: mean win rate"),
                       ("recent_std_win_rate", "Last 5 checks: spread (SD)")]:
        print(f"{label:<30} {comparison['dqn'][key]:>8.1%} {comparison['double_dqn'][key]:>12.1%}")
    print("DQN recent wins:", comparison["dqn"]["recent_win_rates"])
    print("Double DQN recent wins:", comparison["double_dqn"]["recent_win_rates"])
    print("Less variation alone does not mean stronger play. Compare win rates too.")
    print(comparison["scope"])
    (comparison_folder / "comparison.json").write_text(json.dumps(comparison, indent=2))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chart skipped: matplotlib is not installed.")
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(reference_episodes, [100*x for x in reference_rates], "o-", label="DQN")
        ax.plot(current_episodes, [100*x for x in current_rates], "o-", label="Double DQN")
        ax.set(xlabel="Training episodes", ylabel="Validation win rate (%)", ylim=(-2, 102),
               title="Fixed rook endgame vs random opponent (50 games per check)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(comparison_folder / "comparison.png", dpi=160)
        plt.show()
    print("Comparison saved in:", comparison_folder)
