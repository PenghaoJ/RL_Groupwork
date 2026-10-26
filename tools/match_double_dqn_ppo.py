"""Play the selected Double DQN and PPO checkpoints with paired side swaps."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
import torch
from notebook_tools import replay_episode
from rules import Game, action_move

MODELS = {
    "double_dqn": {
        "name": "Old Double DQN (best episode 200)",
        "folder": "xiangqi_lab/models/local_match_double_dqn",
        "checkpoint": "best.pt",
        "sha256": "b616b5c8fdb10587a8b9054484348ee0db118b02cf2b4e6e5e54cba9d90d7379",
    },
    "ppo42": {
        "name": "PPO seed 42 (best episode 700)",
        "folder": "xiangqi_lab/models/local_ppo_rook20_seed42",
        "checkpoint": "weights.zip",
        "sha256": "0f06447b8c97fd433b425a65f2427f055ccda65ab79fe173fd7f3e2a98ee3876",
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_policy(key, item):
    folder = ROOT / item["folder"]
    assert sha(folder / item["checkpoint"]) == item["sha256"]
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(f"match_{key}", folder / "policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Policy(str(folder / item["checkpoint"]), device="cpu")


def tally(rows):
    counts = Counter(r["result"] for r in rows)
    return {"games": len(rows), **{k: counts[k] for k in
            ("double_dqn_win", "ppo42_win", "draw", "ply_limit")}}


def report(output, summary, rows):
    table = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row[k]))}</td>" for k in
        ("board", "red", "black", "result", "reason", "plies")) +
        f'<td><a href="{escape(row["replay"])}">JSON replay</a></td></tr>'
        for row in rows)
    totals = summary["totals"]
    output.joinpath("report.html").write_text(f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Double DQN vs PPO</title>
<style>body{{font:17px/1.6 system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#243b36;background:#f7f8f5}}
h1{{font-size:32px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}.score{{font-size:24px;background:#e5efe8;padding:20px;border-radius:12px}}code{{font-size:14px}}</style>
<h1>Old Double DQN vs PPO seed 42</h1>
<p class="score">Double DQN: {totals['double_dqn_win']} wins · PPO: {totals['ppo42_win']} wins · Draws: {totals['draw']} · Move limits: {totals['ply_limit']}</p>
<p>10 existing test boards × 2 player assignments = 20 games. Each starting board stays the same; the models exchange Red and Black. Both use deterministic moves. No training. Checkpoint files and loaded tensors remained unchanged.</p>
<p>Limit: 160 plies (80 full turns). Threefold repetition or 120 plies without a capture is a draw under the project's research rules. Move-limit games are reported separately.</p>
<p>Both models were trained as Red; Black uses the tested color and board-view transform. This does not add defensive training. These boards have been evaluated before; this is an exploratory comparison of these checkpoints, not a fresh blind test or an algorithm-wide ranking.</p>
<p>Double DQN best episode: 200. PPO seed 42 best episode: 700. PPO seed 42 was the preview choice before this match and had the highest delivered test score among three PPO seeds; no further selection was made from these match results.</p>
<p><a href="summary.json">Audit and totals</a> · <a href="games.csv">Game table</a> · <a href="plan.json">Saved plan</a></p>
<table><thead><tr><th>Board</th><th>Red</th><th>Black</th><th>Result</th><th>Reason</th><th>Plies</th><th>Replay</th></tr></thead><tbody>{table}</tbody></table>
<p>Use Load game on Xiangqi Lab to open any JSON replay.</p></html>""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    bank_path = ROOT / "output/dqn/UNSEEN10_FOUR_20261025/boards.json"
    bank = json.loads(bank_path.read_text(encoding="utf-8-sig"))
    assert len(bank["boards"]) == 10
    jobs = [dict(board=b["id"], group=b["group"], fen=b["fen"], red=r, black=k)
            for b in bank["boards"]
            for r, k in (("double_dqn", "ppo42"), ("ppo42", "double_dqn"))]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), training=False,
                deterministic=True, max_plies=160, models=MODELS, jobs=jobs,
                bank_sha256=sha(bank_path), rules_sha256=sha(ROOT / "xiangqi_lab/rules.py"),
                seeds="No stochastic opponent or exploration; repeat seeds would duplicate games.",
                black_view="Rotate 180 degrees and swap colors, including repetition history.")
    save(output / "plan.json", plan)
    policies = {k: load_policy(k, item) for k, item in MODELS.items()}
    networks = {k: p.model if k == "ppo42" else p.policy.model for k, p in policies.items()}
    for network in networks.values():
        network.eval()
        network.requires_grad_(False)
    before = {k: {n: t.detach().clone() for n, t in net.state_dict().items()}
              for k, net in networks.items()}
    rows = []
    with torch.inference_mode():
        for index, job in enumerate(jobs, 1):
            game = Game(job["fen"], max_plies=160)
            assert game.outcome() is None, job["board"]
            while game.outcome() is None:
                actor = job[game.side]
                observation = game.observation()
                result = policies[actor].select_move(observation)
                action = result["action_id"] if isinstance(result, dict) else int(result)
                assert action in observation["legal_action_ids"]
                game.step(action_move(action), actor=MODELS[actor]["name"])
            outcome = game.outcome()
            episode = game.export()
            assert replay_episode(episode).outcome() == outcome
            replay = f"episodes/{job['board']}_red_{job['red']}.json"
            save(output / replay, episode)
            winner = job[outcome["winner"]] if outcome["winner"] else None
            result = winner + "_win" if winner else "ply_limit" if outcome["truncated"] else "draw"
            row = dict(board=job["board"], group=job["group"], red=job["red"], black=job["black"],
                       result=result, winner_side=outcome["winner"], reason=outcome["reason"],
                       plies=len(game.history), replay=replay, final_fen=game.fen)
            rows.append(row)
            print(f"{index:02}/20 {job['board']} Red={job['red']}: {result}, {outcome['reason']}, {len(game.history)} plies", flush=True)
    for key, item in MODELS.items():
        assert sha(ROOT / item["folder"] / item["checkpoint"]) == item["sha256"]
        assert all(torch.equal(t, networks[key].state_dict()[n]) for n, t in before[key].items())
    summary = dict(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(),
                   training=False, weights_unchanged=True, replay_validation=True,
                   totals=tally(rows), reasons=dict(Counter(r["reason"] for r in rows)),
                   groups={g: tally([r for r in rows if r["group"] == g]) for g in ("rook", "cannon", "horse")},
                   assignments={k: tally([r for r in rows if r["red"] == k]) for k in MODELS},
                   total_plies=sum(r["plies"] for r in rows), models=MODELS,
                   torch_version=torch.__version__, bank_sha256=plan["bank_sha256"])
    save(output / "summary.json", summary)
    with (output / "games.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report(output, summary, rows)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
