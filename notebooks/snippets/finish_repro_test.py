# @title 17. Test both best models on 200 new seeds
# Freeze both models, test them on the same seeds, and save the results.
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
from IPython.display import HTML, display

test_summary = json.loads((REPRO_DIR / "summary.json").read_text())
assert test_summary["stop_reason"] in {"early_stopping", "max_episodes"}
assert test_summary["minimum_reached"], "Finish the original training budget first."
assert hashlib.sha256(RETAINED_DDQN.read_bytes()).hexdigest() == RETAINED_DDQN_SHA
test_dqn = repro_training.torch.load(REPRO_DIR / "best.pt", map_location="cpu", weights_only=True)
assert test_dqn["episode"] == test_summary["best_episode"]
assert test_dqn["config"] == vars(repro_config)
test_dqn_sha = hashlib.sha256((REPRO_DIR / "best.pt").read_bytes()).hexdigest()
TEST_DIR = Path("/content/xiangqi_output/comparisons/DQN_vs_DoubleDQN_TEST200_20261022")
print("Training is finished. Both best models are now frozen.", flush=True)
print("DQN:", REPRO_NAME, "| best episode:", test_dqn["episode"], "| SHA256:", test_dqn_sha, flush=True)
print("Double DQN: original run | best episode: 200 | SHA256:", RETAINED_DDQN_SHA, flush=True)

# EVALUATOR_SOURCE is added by the notebook preparation script.
test_module_path = Path(repro_training.__file__).with_name("compare_rook_checkpoints.py")
test_module_path.write_text(EVALUATOR_SOURCE, encoding="utf-8")
test_spec = importlib.util.spec_from_file_location("frozen_rook_comparison", test_module_path)
test_module = importlib.util.module_from_spec(test_spec)
test_spec.loader.exec_module(test_module)
existing_report = TEST_DIR / "comparison.json"
if existing_report.is_file():
    comparison_result = json.loads(existing_report.read_text())
    assert comparison_result["status"] == "complete", "Existing test is incomplete; inspect it before continuing."
    assert comparison_result["games_per_model"] == 200 and comparison_result["split"] == test_module.SPLIT
    assert comparison_result["results"]["dqn"]["checkpoint_sha256"] == test_dqn_sha
    assert comparison_result["results"]["double_dqn"]["checkpoint_sha256"] == RETAINED_DDQN_SHA
    assert all(row["weights_unchanged"] for row in comparison_result["results"].values())
    print("Loaded the completed test. No games were repeated.", flush=True)
else:
    comparison_result = test_module.compare(REPRO_DIR / "best.pt", RETAINED_DDQN, test_dqn_sha, TEST_DIR)

test_labels = {"dqn": "DQN R2 reproduction", "double_dqn": "Double DQN original"}
print("\nModel                       Wins  Losses  Draws  Limit  Win rate", flush=True)
for key, label in test_labels.items():
    row = comparison_result["results"][key]
    print(f"{label:27} {row['wins']:4} {row['losses']:7} {row['draws']:6} {row['truncations']:6} {row['win_rate']:9.1%}", flush=True)

readme = ["# DQN and Double DQN: 200-game test", "",
          "DQN is a new reproduction of the second-run settings. Double DQN is the preserved original model.",
          "Both best models were chosen using validation before this test. No weights changed in testing.", "",
          "| Model | Best episode | Wins | Losses | Draws | Move limit | Win rate |",
          "|---|---:|---:|---:|---:|---:|---:|"]
for key, label in test_labels.items():
    row = comparison_result["results"][key]
    readme.append(f"| {label} | {row['best_episode']} | {row['wins']} | {row['losses']} | {row['draws']} | {row['truncations']} | {row['win_rate']:.1%} |")
readme += ["", "The test uses one fixed rook endgame, Red as the learner, a uniform random legal-move opponent,",
           "a 160-ply limit, no exploration, and the same 200 new opponent seeds for both models.",
           "Draws and move-limit truncations are reported separately. Win rate is wins divided by 200.", "",
           "This measures performance on new opponent seeds, not on unseen boards or against Pikafish.",
           "One training seed per algorithm does not establish which algorithm is generally more reliable.", "",
           "Models are backed up separately. Tests contain per-game CSV files and comparison.json."]
(TEST_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 3.6))
starts = [0, 0]
for key, label, colour in (("wins", "Win", "#3b8065"), ("losses", "Loss", "#c85b50"),
                           ("draws", "Draw", "#a8b3b5"), ("truncations", "Move limit", "#d8ac48")):
    values = [comparison_result["results"][name][key] for name in test_labels]
    bars = ax.barh(list(test_labels.values()), values, left=starts, label=label, color=colour)
    for i, value in enumerate(values):
        if value >= 12:
            ax.text(starts[i] + value / 2, i, str(value), ha="center", va="center", color="white" if key == "wins" else "#222222")
    starts = [a + b for a, b in zip(starts, values)]
ax.set_xlim(0, 200)
ax.invert_yaxis()
ax.set_xlabel("Games (same 200 new seeds)")
ax.set_title("Frozen models: rook endgame vs random opponent")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False)
fig.tight_layout()
fig.savefig(TEST_DIR / "comparison.png", dpi=160, bbox_inches="tight")
plt.show()

bundle = TEST_DIR / "Xiangqi_DQN_Repro_vs_DoubleDQN_Test200_20261022_Results.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in ("comparison.json", "dqn_games.csv", "double_dqn_games.csv", "README.md", "comparison.png"):
        archive.write(TEST_DIR / name, "tests/" + name)
    archive.writestr("README.md", "\n".join(readme) + "\n")
bundle_bytes = bundle.read_bytes()
bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
display(HTML(f'<p><b>Complete: 200 games per model. No training during testing.</b></p>'
             f'<a download="{bundle.name}" href="data:application/zip;base64,{base64.b64encode(bundle_bytes).decode()}" '
             f'data-sha256="{bundle_sha}">Save test results</a>'
             f'<p>ZIP SHA256: {bundle_sha}</p>'))
print("Complete bundle:", bundle.name, "| SHA256:", bundle_sha, flush=True)
