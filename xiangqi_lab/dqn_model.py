"""DQN network and stateless website policy; PyTorch is an optional dependency."""
from functools import lru_cache

import torch
from torch import nn

MODEL_SCHEMA = "xiangqi-lab.dqn.v1"
PIECE_ORDER = "KAEHRCPkaehrcp"
ACTION_COUNT = 8100


@lru_cache(maxsize=32768)
def position_tokens(key):
    """Encode a position key without changing orientation or action numbering."""
    board, side = key.split()[:2]
    tokens = []
    for char in board:
        if char == "/":
            continue
        if char in "123456789":
            tokens.extend([0] * int(char))
        else:
            tokens.append(PIECE_ORDER.index(char) + 1)
    if len(tokens) != 90 or side not in ("w", "b", "r"):
        raise ValueError("Invalid position key.")
    return bytes(tokens), int(side == "b")


def encode_state(observation):
    """Keep every repetition entry, not only the current position's count.

    Values are standard Python types so replay can be saved safely with torch.
    The external episode time limit is not an input to the continuing-game value.
    """
    key = " ".join(observation["fen"].split()[:2])
    board, side = position_tokens(key)
    history = tuple((*position_tokens(k), int(count))
                    for k, count in sorted(observation["repetition_counts"].items()) if count > 0)
    if not history:
        raise ValueError("The observation must include repetition counts.")
    return board, side, float(observation["halfmove_clock"]) / 120.0, history


def batch_states(states, device):
    boards, metadata, history_boards, history_meta, owners = [], [], [], [], []
    for owner, (board, side, clock, history) in enumerate(states):
        boards.append(list(board))
        metadata.append([side, clock, len(history) / 160.0])
        for old_board, old_side, count in history:
            history_boards.append(list(old_board))
            history_meta.append([old_side, count / 3.0])
            owners.append(owner)
    return (
        torch.tensor(boards, dtype=torch.long, device=device),
        torch.tensor(metadata, dtype=torch.float32, device=device),
        torch.tensor(history_boards, dtype=torch.long, device=device),
        torch.tensor(history_meta, dtype=torch.float32, device=device),
        torch.tensor(owners, dtype=torch.long, device=device),
    )


class QNetwork(nn.Module):
    """Score all action IDs, using the current board and the full repetition map."""
    def __init__(self, hidden=64, embedding=4, history_hidden=16):
        super().__init__()
        self.config = {"hidden": hidden, "embedding": embedding, "history_hidden": history_hidden}
        self.pieces = nn.Embedding(15, embedding, padding_idx=0)
        self.current = nn.Sequential(nn.Linear(90 * embedding + 3, hidden), nn.ReLU())
        self.history = nn.Sequential(nn.Linear(90 * embedding + 2, history_hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden + history_hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, ACTION_COUNT))

    def forward(self, batch):
        boards, metadata, old_boards, old_meta, owners = batch
        current = self.current(torch.cat((self.pieces(boards).flatten(1), metadata), dim=1))
        old = self.history(torch.cat((self.pieces(old_boards).flatten(1), old_meta), dim=1))
        # Each complete repetition map is pooled as an unordered set.
        pooled = old.new_zeros((len(boards), old.shape[1])).index_add(0, owners, old)
        count = torch.bincount(owners, minlength=len(boards)).clamp_min(1).sqrt().unsqueeze(1)
        return self.head(torch.cat((current, pooled / count), dim=1))


@torch.no_grad()
def greedy_action(model, state, legal_ids, device):
    if not legal_ids:
        raise ValueError("No legal action is available.")
    q = model(batch_states([state], device))[0]
    legal = torch.tensor(legal_ids, dtype=torch.long, device=device)
    return int(legal[q[legal].argmax()].item())


class Policy:
    """Load an exported checkpoint once and serve the website's Python protocol."""
    def __init__(self, checkpoint=None, device="cpu"):
        if not checkpoint:
            raise ValueError("A trained DQN checkpoint is required.")
        torch.set_num_threads(2)
        self.device = torch.device(device)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("schema") != MODEL_SCHEMA:
            raise ValueError("Unsupported DQN checkpoint format.")
        self.model = QNetwork(**saved["network_config"]).to(self.device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()

    def select_move(self, observation, time_budget_ms=800):
        return {"action_id": greedy_action(self.model, encode_state(observation),
                                           observation["legal_action_ids"], self.device)}
