"""Fixed-opponent DQN and Double DQN with validation and early stopping."""
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import random
import sys
import time
import zipfile

import torch
from torch import nn

from dqn_model import MODEL_SCHEMA, QNetwork, batch_states, encode_state, greedy_action
from rules import Game, PRESETS, action_id, action_move

ENDGAMES = {p["id"]: p for p in PRESETS if p["id"] in ("rook", "cannon", "horse")}


class FixedOpponentEnv:
    """One learner decision and (if needed) a fixed-policy opponent reply."""
    def __init__(self, max_plies=160, learner="red", opponent=None):
        if learner not in ("red", "black") or type(max_plies) is not int or not 4 <= max_plies <= 1000:
            raise ValueError("Use red/black and a ply limit from 4 to 1000.")
        self.max_plies, self.learner = max_plies, learner
        self.opponent = opponent
        self.game = None

    def _reply(self):
        move = (self.opponent(self.game.observation(), self.rng) if self.opponent
                else self.rng.choice(self.game.legal))
        self.game.step(move, actor="fixed-opponent")

    def reset(self, fen, seed=0):
        self.rng = random.Random(seed)
        self.game = Game(fen, self.max_plies)
        opening = int(self.game.side != self.learner)
        # Align the limit to the learner's next turn, so a time-limit transition
        # can bootstrap from a learner state. Never exceed the requested cap.
        self.game.max_plies = self.max_plies - ((self.max_plies - opening) % 2)
        if opening and self.game.outcome() is None:
            self._reply()
        return self.game.observation(), {"outcome": self.game.outcome()}

    def step(self, action):
        if self.game is None or self.game.outcome() is not None:
            raise ValueError("Reset before moving in a new episode.")
        if self.game.side != self.learner:
            raise RuntimeError("A training step must start on the learner's turn.")
        before = len(self.game.history)
        move = action_move(action) if type(action) is int else action
        self.game.step(move, actor="dqn")
        if self.game.outcome() is None:
            self._reply()
        outcome = self.game.outcome()
        winner = outcome["winner"] if outcome else None
        reward = (1.0 if winner == self.learner else -1.0) if winner else 0.0
        return (self.game.observation(), reward,
                bool(outcome and outcome["terminated"]), bool(outcome and outcome["truncated"]),
                {"outcome": outcome, "learner": self.learner,
                 "environment_plies": len(self.game.history) - before})


@dataclass
class TrainConfig:
    preset: str = "rook"
    learner: str = "red"
    min_episodes: int = 1000
    max_episodes: int = 3000
    max_plies: int = 160
    eval_every: int = 100
    eval_games: int = 50
    patience: int = 5
    min_delta: float = 0.02
    seed: int = 42
    learning_rate: float = 0.0003
    gamma: float = 0.99
    replay_size: int = 10000
    batch_size: int = 64
    warmup_steps: int = 512
    train_every: int = 4
    target_every: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 20000
    device: str = "auto"
    algorithm: str = "dqn"
    replay_sampling: str = "uniform"
    terminal_mix: float = 0.5

    def validate(self):
        if self.algorithm not in ("dqn", "double_dqn"):
            raise ValueError("Choose dqn or double_dqn.")
        if self.replay_sampling not in ("uniform", "terminal_mixture"):
            raise ValueError("Choose uniform or terminal_mixture replay sampling.")
        if not math.isfinite(self.terminal_mix) or not 0 <= self.terminal_mix < 1:
            raise ValueError("terminal_mix must be finite and in [0, 1).")
        if self.preset not in ENDGAMES or self.learner not in ("red", "black"):
            raise ValueError("Unknown preset or learner colour.")
        for name in ("min_episodes", "max_episodes", "max_plies", "eval_every", "eval_games",
                     "patience", "replay_size", "batch_size", "warmup_steps", "train_every",
                     "target_every", "epsilon_decay_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer.")
        if self.max_episodes < self.min_episodes:
            raise ValueError("MAX_EPISODES must be at least MIN_EPISODES.")
        if not 4 <= self.max_plies <= 1000 or self.replay_size < self.batch_size:
            raise ValueError("Invalid ply limit or replay size.")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Invalid epsilon or discount factor.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite.")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Minimum improvement must be finite and non-negative.")


@dataclass
class EarlyStopping:
    min_episodes: int
    patience: int
    min_delta: float
    best_score: float = -math.inf
    reference_score: float = -math.inf
    bad_evals: int = 0
    monitoring_started: bool = False

    def update(self, episode, score):
        if not math.isfinite(score):
            raise ValueError("Validation score must be finite.")
        improved = score > self.best_score
        if improved:
            self.best_score = score
        if episode < self.min_episodes:
            self.bad_evals = 0
        elif not self.monitoring_started:
            self.monitoring_started = True
            self.reference_score = self.best_score
            self.bad_evals = 0
        elif score > self.reference_score + self.min_delta:
            self.reference_score = score
            self.bad_evals = 0
        else:
            self.bad_evals += 1
        return improved, self.monitoring_started and self.bad_evals >= self.patience


@torch.no_grad()
def legal_next_values(target, next_states, legal_ids, terminated, device, online=None):
    """Double DQN selects with online and evaluates with target, using legal moves."""
    values = torch.zeros(len(next_states), device=device)
    live = [i for i, terminal in enumerate(terminated) if not terminal]
    if live:
        batch = batch_states([next_states[i] for i in live], device)
        scores = target(batch)
        selection = online(batch) if online is not None else scores
        for row, index in enumerate(live):
            if not legal_ids[index]:
                raise ValueError("A nonterminal transition needs legal next actions.")
            legal = torch.tensor(legal_ids[index], dtype=torch.long, device=device)
            chosen = legal[selection[row, legal].argmax()]
            values[index] = scores[row, chosen]
    return values


def transition_group(transition):
    """Time-limit truncations are live transitions, not terminal draws."""
    if not transition[4]:
        return "live"
    return "win" if transition[2] > 0 else "loss" if transition[2] < 0 else "draw"


def sample_replay(replay, rng, config):
    """Mix uniform replay with terminal groups; correct by exact sampling odds."""
    items = list(replay)
    groups = {name: [] for name in ("win", "loss", "draw")}
    for index, item in enumerate(items):
        group = transition_group(item)
        if group != "live":
            groups[group].append(index)
    active = [indices for indices in groups.values() if indices]
    alpha = config.terminal_mix if config.replay_sampling == "terminal_mixture" and active else 0.0
    counts = {name: len(indices) for name, indices in groups.items()}
    counts["live"] = len(items) - sum(counts.values())
    if not alpha:
        # Keep the original uniform path and RNG sequence exactly unchanged.
        batch = rng.sample(items, config.batch_size)
        weights = [1.0] * len(batch)
    else:
        batch, weights = [], []
        for _ in range(config.batch_size):
            index = rng.choice(rng.choice(active)) if rng.random() < alpha else rng.randrange(len(items))
            item = items[index]
            group = transition_group(item)
            probability = (1 - alpha) / len(items)
            if group != "live":
                probability += alpha / (len(active) * counts[group])
            batch.append(item)
            weights.append(1.0 / (len(items) * probability))
    selected = {name: sum(transition_group(item) == name for item in batch) for name in counts}
    return batch, weights, {"pool": counts, "selected": selected, "mixture_active": bool(alpha)}


def optimize(model, target, optimizer, replay, rng, config, device, sampling_stats=None):
    batch, weights, sampled = sample_replay(replay, rng, config)
    states, actions, rewards, next_states, terminals, next_ids = zip(*batch)
    predicted = model(batch_states(states, device)).gather(
        1, torch.tensor(actions, device=device).unsqueeze(1)).squeeze(1)
    expected = torch.tensor(rewards, device=device) + config.gamma * legal_next_values(
        target, next_states, next_ids, terminals, device,
        online=model if config.algorithm == "double_dqn" else None)
    if sampled["mixture_active"]:
        loss = (nn.functional.smooth_l1_loss(predicted, expected, reduction="none") *
                torch.tensor(weights, dtype=predicted.dtype, device=device)).mean()
    else:
        loss = nn.functional.smooth_l1_loss(predicted, expected)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite DQN loss; the run was stopped.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    if sampling_stats is not None:
        sampling_stats["batches"] = sampling_stats.get("batches", 0) + 1
        sampling_stats["mixture_batches"] = sampling_stats.get("mixture_batches", 0) + int(sampled["mixture_active"])
        for name, count in sampled["selected"].items():
            key = "sampled_" + name
            sampling_stats[key] = sampling_stats.get(key, 0) + count
        sampling_stats["last_pool"] = sampled["pool"]
        sampling_stats["last_weight_min"] = min(weights)
        sampling_stats["last_weight_max"] = max(weights)
    return float(loss.item())


@torch.no_grad()
def evaluate(model, config, device, split="validation", games=None, positions=None):
    """Fixed seed bank, fresh envs, no exploration or training-state mutation."""
    count = games or config.eval_games
    if positions is not None and (not positions or count % len(positions)):
        raise ValueError("Validation games must cover every board equally.")
    stats = {"wins": 0, "losses": 0, "draws": 0, "truncations": 0, "environment_plies": 0}
    per_board = {p["id"]: {key: 0 for key in ("wins", "losses", "draws", "truncations")} for p in positions or []}
    previous_mode = model.training if model is not None else False
    if model is not None:
        model.eval()
    try:
        for index in range(count):
            env = FixedOpponentEnv(config.max_plies, config.learner)
            position = positions[index % len(positions)] if positions is not None else ENDGAMES[config.preset]
            seed = (f"{split}:{position['id']}:opponent:{config.seed}:{index // len(positions)}"
                    if positions is not None else f"{split}:opponent:{config.seed}:{index}")
            obs, info = env.reset(position["fen"], seed)
            policy_rng = random.Random(f"{split}:random-policy:{config.seed}:{index}")
            while info["outcome"] is None:
                action = (policy_rng.choice(obs["legal_action_ids"]) if model is None
                          else greedy_action(model, encode_state(obs), obs["legal_action_ids"], device))
                obs, reward, terminated, truncated, info = env.step(action)
            result = info["outcome"]
            key = ("truncations" if result["truncated"] else "draws" if result["winner"] is None
                   else "wins" if result["winner"] == config.learner else "losses")
            stats[key] += 1
            if positions is not None:
                per_board[position["id"]][key] += 1
            stats["environment_plies"] += len(env.game.history)
    finally:
        if model is not None:
            model.train(previous_mode)
    details = {"per_board": per_board} if positions is not None else {}
    return {**stats, **details, "games": count, "split": split,
            "score": (stats["wins"] - stats["losses"]) / count,
            "win_rate": stats["wins"] / count}


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _append_csv(path, row):
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(config, run_dir, resume=False, source_hashes=None, board_bank=None, on_checkpoint=None):
    """Run explicitly. No training starts when this module is imported."""
    config.validate()
    if board_bank is not None and config.eval_games % len(board_bank.validation):
        raise ValueError("Validation games must cover every board equally.")
    validation_positions = board_bank.validation if board_bank is not None else None
    run_dir = Path(run_dir).resolve()
    if resume:
        if not (run_dir / "last.pt").is_file():
            raise FileNotFoundError("No last.pt checkpoint in this run folder.")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Use a new run folder, or explicitly resume an existing run.")
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available()
                          else "cpu" if config.device == "auto" else config.device)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    replay_rng = random.Random(config.seed + 1)
    model, target = QNetwork().to(device), QNetwork().to(device)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    replay = deque(maxlen=config.replay_size)
    stopper = EarlyStopping(config.min_episodes, config.patience, config.min_delta)
    steps = updates = total_plies = completed = 0
    best_episode = None
    last_loss = None
    last_validation = None
    sampling_stats = {}
    if resume:
        saved = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if saved.get("schema") != MODEL_SCHEMA or saved.get("kind") != "training":
            raise ValueError("Unsupported training checkpoint.")
        for key, value in asdict(config).items():
            defaults = {"algorithm": "dqn", "replay_sampling": "uniform", "terminal_mix": 0.5}
            previous = saved["config"].get(key, defaults.get(key))
            if key not in ("max_episodes", "device") and previous != value:
                raise ValueError("Resume configuration differs: " + key)
        if saved.get("source_hashes", {}) != (source_hashes or {}):
            raise ValueError("Resume requires the same source snapshot.")
        if saved.get("board_bank_sha256") != (board_bank.sha256 if board_bank is not None else None):
            raise ValueError("Resume requires the same training and validation board bank.")
        model.load_state_dict(saved["model"])
        target.load_state_dict(saved["target"])
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        replay.extend(saved["replay"])
        rng.setstate(saved["rng"])
        replay_rng.setstate(saved["replay_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        stopper = EarlyStopping(**saved["early_stopping"])
        completed, steps, updates, total_plies = (saved[k] for k in ("episode", "steps", "updates", "total_plies"))
        best_episode, last_loss, last_validation = saved["best_episode"], saved["last_loss"], saved["validation"]
        sampling_stats = saved.get("sampling_stats", {})
        if completed >= config.max_episodes:
            raise ValueError("The saved run already reached MAX_EPISODES. Increase it to resume.")
        if not (run_dir / "best.pt").is_file():
            raise FileNotFoundError("Resume needs the original best.pt beside last.pt.")
        # CSV rows written after the last completed checkpoint may be from an
        # interrupted episode range; remove those rows before replaying that range.
        for name in ("episodes.csv", "validation.csv"):
            path = run_dir / name
            if path.exists():
                with path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    fields, rows = reader.fieldnames, list(reader)
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(row for row in rows if int(row["episode"]) <= completed)
    else:
        print("Checking the random-policy baseline...", flush=True)
        _write_json(run_dir / "baseline.json", evaluate(None, config, device, positions=validation_positions))

    if board_bank is not None:
        _write_json(run_dir / "board_bank.json", board_bank.data)

    _write_json(run_dir / "config.json", {**asdict(config), "actual_device": str(device),
                "source_hashes": source_hashes or {}, "torch_version": str(torch.__version__),
                "python_version": sys.version.split()[0], "opponent": "uniform-legal-random-v1",
                "board_bank_sha256": board_bank.sha256 if board_bank is not None else None,
                "validation_scope": "separate validation starting boards; fixed seeds; no test-board selection"
                if board_bank is not None else "same preset; separate fixed random seeds; no unseen-board claim"})
    started = time.monotonic()
    stop_reason = "max_episodes"

    def checkpoint(kind, validation):
        payload = {"schema": MODEL_SCHEMA, "kind": kind, "network_config": model.config,
                   "model": model.state_dict(), "config": asdict(config), "episode": completed,
                   "steps": steps, "updates": updates, "total_plies": total_plies,
                   "validation": validation, "source_hashes": source_hashes or {}}
        if board_bank is not None:
            payload["board_bank_sha256"] = board_bank.sha256
        if kind == "training":
            payload.update(target=target.state_dict(), optimizer=optimizer.state_dict(), replay=list(replay),
                           rng=rng.getstate(), replay_rng=replay_rng.getstate(), torch_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                           early_stopping=asdict(stopper), best_episode=best_episode, last_loss=last_loss,
                           sampling_stats=sampling_stats)
        return payload

    print(f"Algorithm: {config.algorithm} | Device: {device} | At least {config.min_episodes} training episodes | Folder: {run_dir}", flush=True)
    print("Early-stop patience starts at the first validation after the minimum episode budget.", flush=True)
    try:
        for episode in range(completed + 1, config.max_episodes + 1):
            env = FixedOpponentEnv(config.max_plies, config.learner)
            position = board_bank.training_position(episode, config.seed) if board_bank is not None else ENDGAMES[config.preset]
            obs, info = env.reset(position["fen"], f"train:opponent:{config.seed}:{episode}")
            if info["outcome"] is not None:
                raise ValueError("The training preset ended before the learner could move.")
            total_plies += len(env.game.history)
            episode_steps, reward, epsilon = 0, 0.0, config.epsilon_start
            while True:
                state = encode_state(obs)
                fraction = min(1.0, steps / config.epsilon_decay_steps)
                epsilon = config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
                action = (rng.choice(obs["legal_action_ids"]) if rng.random() < epsilon else
                          greedy_action(model, state, obs["legal_action_ids"], device))
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = encode_state(next_obs)
                # A truncated game retains next-state legal IDs for bootstrapping.
                replay.append((state, action, reward, next_state, terminated,
                               tuple(next_obs["legal_action_ids"]) if not terminated else ()))
                steps += 1
                episode_steps += 1
                total_plies += info["environment_plies"]
                if steps >= config.warmup_steps and len(replay) >= config.batch_size and steps % config.train_every == 0:
                    last_loss = optimize(model, target, optimizer, replay, replay_rng, config, device, sampling_stats)
                    updates += 1
                    if updates % config.target_every == 0:
                        target.load_state_dict(model.state_dict())
                obs = next_obs
                if terminated or truncated:
                    break
            completed = episode
            board_details = {"board_id": position["id"], "initial_fen": position["fen"]} if board_bank is not None else {}
            _append_csv(run_dir / "episodes.csv", {"episode": episode, "learner_steps": steps,
                        "environment_plies": total_plies, "episode_plies": len(env.game.history),
                        "reward": reward, "reason": info["outcome"]["reason"],
                        "epsilon": epsilon, "loss": last_loss, "updates": updates, **board_details})
            if episode % 25 == 0:
                print(f"Episode {episode}/{config.max_episodes} | steps {steps} | epsilon {epsilon:.3f}", flush=True)
            if episode % config.eval_every == 0 or episode == config.min_episodes or episode == config.max_episodes:
                last_validation = evaluate(model, config, device, positions=validation_positions)
                improved, stop = stopper.update(episode, last_validation["score"])
                if improved:
                    best_episode = episode
                    _save_torch(run_dir / "best.pt", checkpoint("policy", last_validation))
                _append_csv(run_dir / "validation.csv", {"episode": episode,
                            **{k: v for k, v in last_validation.items() if k != "per_board"},
                            "bad_evals": stopper.bad_evals, "best_episode": best_episode})
                if board_bank is not None:
                    _write_json(run_dir / f"validation_ep{episode:04d}.json", last_validation)
                _write_json(run_dir / "example_game.json", env.game.export())
                _save_torch(run_dir / "last.pt", checkpoint("training", last_validation))
                _write_json(run_dir / "sampling.json", {"episode": completed, **sampling_stats,
                            "mode": config.replay_sampling, "terminal_mix": config.terminal_mix,
                            "importance_weight": "1 / (buffer_size * item_probability); no normalization"})
                if on_checkpoint is not None:
                    on_checkpoint(run_dir, improved)
                print(f"Validation {episode}: W/L/D/T = {last_validation['wins']}/{last_validation['losses']}/"
                      f"{last_validation['draws']}/{last_validation['truncations']} | score {last_validation['score']:.3f} | "
                      f"patience {stopper.bad_evals}/{config.patience}", flush=True)
                if stop:
                    stop_reason = "early_stopping"
                    break
    except KeyboardInterrupt:
        # last.pt refers only to a completed validation checkpoint. Preserve it;
        # do not label a partially played episode as completed training.
        stop_reason = "interrupted"
        print("Interrupted. Resume from the latest completed last.pt checkpoint.", flush=True)
    summary = {"algorithm": config.algorithm,
               "episodes_completed": completed, "minimum_reached": completed >= config.min_episodes,
               "learner_steps": steps, "environment_plies": total_plies, "updates": updates,
               "stop_reason": stop_reason, "best_episode": best_episode,
               "best_validation_score": stopper.best_score if math.isfinite(stopper.best_score) else None,
               "seconds_this_session": time.monotonic() - started, "run_dir": str(run_dir)}
    _write_json(run_dir / "summary.json", summary)
    if best_episode is not None:
        best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(best["model"])
    print(json.dumps(summary, indent=2), flush=True)
    return {"model": model, "config": config, "device": device, "summary": summary, "run_dir": run_dir}


def export_policy(run_dir, model_id=None, model_name=None):
    """Create a complete model folder ZIP for the website, without replay memory."""
    run_dir = Path(run_dir)
    saved = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    preset = saved["config"]["preset"]
    algorithm = saved["config"].get("algorithm", "dqn")
    label = "Double DQN" if algorithm == "double_dqn" else "DQN"
    folder = model_id if model_id is not None else f"{algorithm}_{preset}"
    if not folder or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in folder):
        raise ValueError("Use a simple portable model ID.")
    output = run_dir / f"{folder}_model.zip"
    manifest = {"schema": "xiangqi-lab.model.v1", "id": model_id or f"custom-{algorithm.replace('_', '-')}-{preset}",
                "name": model_name or f"{label} {preset}", "description": f"Best validation checkpoint, episode {saved['episode']}; trained as {saved['config']['learner']}.",
                "protocol": "python", "entrypoint": "policy.py", "python": "{python}",
                "checkpoint": "best.pt", "device": "cpu", "timeout_seconds": 60}
    readme = (f"# {label} {preset}\n\nTrained as {saved['config']['learner']}, selected at episode {saved['episode']}.\n"
              "Extract this folder under xiangqi_lab/models/. Install requirements.txt in the Python environment "
              "that starts the website, or set model.json's python path to a separate environment with PyTorch. "
              "The portable app's bundled Python does not include PyTorch.\n\n"
              "Use Models > Refresh list > Test before selecting the model. Legal moves are masked for either "
              "colour; playing strength for the untrained colour and unseen boards is not established.\n\n"
              "Validation uses the same fixed starting board with separate opponent seeds. "
              "It is used for checkpoint selection, not a held-out-board generalisation claim.\n")
    if saved.get("board_bank_sha256"):
        readme = readme.replace("Validation uses the same fixed starting board with separate opponent seeds.",
                                "Validation uses separate starting boards and fixed opponent seeds; test boards are excluded.")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(run_dir / "best.pt", f"{folder}/best.pt")
        archive.write(Path(__file__).with_name("dqn_model.py"), f"{folder}/dqn_model.py")
        archive.writestr(f"{folder}/policy.py", "from dqn_model import Policy\n")
        archive.writestr(f"{folder}/model.json", json.dumps(manifest, indent=2))
        archive.writestr(f"{folder}/requirements.txt", "torch>=2.6,<3\n")
        archive.writestr(f"{folder}/README.md", readme)
        for name in ("config.json", "summary.json", "baseline.json", "episodes.csv", "validation.csv", "example_game.json", "board_bank.json", "sampling.json"):
            if (run_dir / name).exists():
                archive.write(run_dir / name, f"{folder}/training_report/{name}")
    return output
