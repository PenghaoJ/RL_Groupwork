"""Check that the terminal replay experiment keeps the original controls."""
import ast
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
import torch
from board_bank import BoardBank
from dqn_training import TrainConfig


def verify():
    path = ROOT / "notebooks/Xiangqi_Rook20_Terminal50_DQN_DoubleDQN_20261025.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in code:
        compile(source, "cell", "exec")
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(code[0]).body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ("SOURCE_FILES", "SOURCE_HASHES", "BOARD_DATA")}
    for name, source in values["SOURCE_FILES"].items():
        assert source == (ROOT / "xiangqi_lab" / name).read_text(encoding="utf-8"), name
        assert hashlib.sha256(source.encode()).hexdigest() == values["SOURCE_HASHES"][name], name
    bank = BoardBank(values["BOARD_DATA"])
    options = {"TrainConfig": TrainConfig}
    exec(code[1], options)
    assert not options["START_TRAINING"]
    base = ROOT / "output/dqn/ROOK20_TERMINAL50_20261025_S42/baselines"
    manifest = json.loads((base / "manifest.json").read_text())
    for key, meta in manifest.items():
        for source in (Path(meta["original_path"]), base / Path(meta["original_path"]).name):
            assert hashlib.sha256(source.read_bytes()).hexdigest() == meta["file_sha256"], key
        with zipfile.ZipFile(source) as archive:
            raw = archive.read(next(n for n in archive.namelist() if n.endswith("/best.pt")))
            assert hashlib.sha256(raw).hexdigest() == meta["checkpoint_sha256"]
            if key.startswith("new_"):
                saved = torch.load(io.BytesIO(raw), weights_only=True)
                assert saved["board_bank_sha256"] == bank.sha256, "Frozen board bank changed"
    for algorithm, config in options["CONFIGS"].items():
        previous = manifest["new_" + algorithm]["training_config"]
        assert {k: asdict(config)[k] for k in previous} == previous, "Original setting changed"
    for name, source in values["SOURCE_FILES"].items():
        if name != "dqn_training.py":
            assert source == (base / "source" / name).read_text(encoding="utf-8"), name
    old = "Xiangqi_Rook20_DQN_DoubleDQN_20261025.ipynb"
    assert (ROOT / "notebooks" / old).read_bytes() == (base / old).read_bytes()
    audit = dict(status="ready", change="terminal mixture alpha=0.5; exact importance weights",
        original_config_unchanged=True, other_five_sources_unchanged=True, four_baselines_verified=True,
        board_bank_sha256=bank.sha256, source_hashes=values["SOURCE_HASHES"], runs=options["RUNS"],
        configs={k: asdict(v) for k, v in options["CONFIGS"].items()}, test_cases_per_model=200,
        test_training=False, notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (base.parent / "preflight.json").write_text(json.dumps(audit, indent=2))
    print("Preflight passed: four backups, all original controls, frozen board bank and old notebook verified.")


if __name__ == "__main__":
    verify()
