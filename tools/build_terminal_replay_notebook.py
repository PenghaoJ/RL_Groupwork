"""Build the single-change terminal replay experiment, preserving the baseline."""
from build_rook20_notebook import build
import hashlib
import json
from pathlib import Path


def build_experiment():
    path = build(terminal_mixture=True)
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(path.read_text(encoding="utf-8"))
    test_path = root / "output/dqn/UNSEEN10_FOUR_20261025/boards.json"
    assert hashlib.sha256(test_path.read_bytes()).hexdigest() == "21965057bc63b1df9d90e6d400d83d0eb2b75bc6f936782c347b5f09590ff045"
    reference = json.loads((test_path.parent / "results/comparison.json").read_text())
    evaluation = (root / "tools/terminal_replay_evaluation.py").read_text(encoding="utf-8")
    extra = f'''
if START_TRAINING and (BASE_DIR / "completed.json").is_file():
    # Tests start only after the training and best-model selection are complete.
    TEST_DATA = {json.loads(test_path.read_text())!r}
    REFERENCE = {reference!r}
    EVALUATION_SOURCE = {evaluation!r}
    exec(compile(EVALUATION_SOURCE, "terminal_replay_evaluation.py", "exec"))
    comparison_zip, comparison = run_frozen_comparison(BASE_DIR, RUNS, TEST_DATA, REFERENCE)
    data = comparison_zip.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    encoded = base64.b64encode(data).decode("ascii")
    display(HTML(f'<p>Both runs and 400 frozen test games complete.</p><a download="{{comparison_zip.name}}" data-sha256="{{digest}}" href="data:application/zip;base64,{{encoded}}">Save models and test results</a>'))
    print("TRAINING AND FROZEN TESTS COMPLETE", flush=True)
'''
    last = notebook["cells"][-1]
    last["source"] = ("".join(last["source"]) + "\n" + extra).splitlines(keepends=True)
    notebook["cells"][-2]["source"] = ["## 3. Train and compare\n\nTrain both models, save their best checkpoints, then test each on the same 200 cases without learning.\n"]
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return path

if __name__ == "__main__":
    build_experiment()
