# @title 16. Train DQN again with the second-run settings
# Train one named DQN run and keep a backup of each new best model.
import base64
import hashlib
import json
from pathlib import Path
import zipfile
from IPython.display import HTML, display
import dqn_training as repro_training

REPRO_NAME = "DQN_R2_REPRO_20261022_S42"
REPRO_DIR = Path("/content/xiangqi_output/runs") / REPRO_NAME
RETAINED_DDQN = Path("/content/xiangqi_output/runs/double_dqn-rook-20261022T033907845632Z/best.pt")
RETAINED_DDQN_SHA = "b616b5c8fdb10587a8b9054484348ee0db118b02cf2b4e6e5e54cba9d90d7379"
assert RETAINED_DDQN.is_file(), "Restore the preserved Double DQN checkpoint first."
assert hashlib.sha256(RETAINED_DDQN.read_bytes()).hexdigest() == RETAINED_DDQN_SHA
assert not REPRO_DIR.exists(), "This run already exists. Do not overwrite it."

repro_config = repro_training.TrainConfig(
    preset="rook", learner="red", min_episodes=1000, max_episodes=3000,
    max_plies=160, eval_every=100, eval_games=50, patience=5, min_delta=0.02,
    seed=42, learning_rate=0.0003, gamma=0.99, replay_size=10000,
    batch_size=64, warmup_steps=512, train_every=4, target_every=250,
    epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100000,
    device="auto", algorithm="dqn",
)
repro_config.validate()
print("Run:", REPRO_NAME, flush=True)
print("Same settings as the second DQN run. Double DQN stays unchanged.", flush=True)
repro_backup_display = display(HTML("Waiting for the first best checkpoint."), display_id=True)


def publish_repro_backup(phase):
    original_zip = repro_training.export_policy(REPRO_DIR)
    saved = repro_training.torch.load(REPRO_DIR / "best.pt", map_location="cpu", weights_only=True)
    label = f"{REPRO_NAME}_BEST_EP{saved['episode']:04d}_{phase}"
    named_zip = REPRO_DIR / f"{label}.zip"
    folder = "dqn_rook_r2_repro_20261022_s42"
    with zipfile.ZipFile(original_zip) as source, zipfile.ZipFile(named_zip, "w", zipfile.ZIP_DEFLATED) as target:
        for entry in source.infolist():
            relative = entry.filename.split("/", 1)[1]
            data = source.read(entry)
            if relative == "model.json":
                manifest = json.loads(data)
                manifest.update(id="custom-dqn-rook-r2-repro-20261022-s42",
                                name="DQN R2 Repro 2026-10-22 (seed 42)")
                data = json.dumps(manifest, indent=2).encode()
            target.writestr(f"{folder}/{relative}", data)
        target.writestr(f"{folder}/experiment.json", json.dumps({
            "name": REPRO_NAME, "type": "reproduction", "phase": phase,
            "reference_run": "rook-20261021T135047973987Z",
            "best_episode": saved["episode"], "config": saved["config"],
            "checkpoint_sha256": hashlib.sha256((REPRO_DIR / "best.pt").read_bytes()).hexdigest(),
            "retained_double_dqn_sha256": RETAINED_DDQN_SHA,
            "test_started": False,
        }, indent=2))
    data = named_zip.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    encoded = base64.b64encode(data).decode("ascii")
    repro_backup_display.update(HTML(
        f'<p><b>{label}</b> | Model saved. No test games yet.</p>'
        f'<a download="{named_zip.name}" href="data:application/zip;base64,{encoded}" '
        f'data-sha256="{digest}">Save model backup</a>'
        f'<p>ZIP SHA256: {digest}</p>'
    ))
    print("Backup ready:", named_zip.name, "| SHA256:", digest, flush=True)
    return named_zip


repro_original_save = repro_training._save_torch


def save_repro_checkpoint(path, value):
    repro_original_save(path, value)
    if Path(path) == REPRO_DIR / "best.pt":
        publish_repro_backup("CHECKPOINT")


repro_training._save_torch = save_repro_checkpoint
try:
    repro_result = repro_training.train(repro_config, REPRO_DIR, source_hashes=SOURCE_HASHES)
finally:
    repro_training._save_torch = repro_original_save
assert hashlib.sha256(RETAINED_DDQN.read_bytes()).hexdigest() == RETAINED_DDQN_SHA
repro_final_zip = publish_repro_backup("FINAL")
print("Training finished. Back up this model before running the separate 200-game test.", flush=True)
