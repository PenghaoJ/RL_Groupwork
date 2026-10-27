"""Encode the shared observation identically in training and website inference."""
from functools import lru_cache
import numpy as np
from gymnasium import spaces

PIECES = 'KAEHRCPkaehrcp'
ACTION_COUNT = 8100
ENCODING_VERSION = 'xiangqi-ppo-history-v1'


@lru_cache(maxsize=8192)
def position_row(key):
    board, side = key.split()
    values = []
    for rank in board.split('/'):
        for ch in rank:
            if ch.isdigit():
                values.extend([0.0] * int(ch))
            else:
                values.append((PIECES.index(ch) + 1) / 14.0)
    if len(values) != 90 or side not in ('w', 'r', 'b'):
        raise ValueError('Invalid repetition position.')
    return tuple(values) + (float(side != 'b'),)


def observation_space(history_capacity):
    return spaces.Dict({
        'board': spaces.Box(0, 1, shape=(14, 10, 9), dtype=np.float32),
        'stats': spaces.Box(0, 1, shape=(6,), dtype=np.float32),
        'history': spaces.Box(0, 1, shape=(history_capacity, 92), dtype=np.float32),
    })


def encode_observation(obs, history_capacity, learner_side=None):
    if obs['plane_order'] != PIECES:
        raise ValueError('Unsupported piece plane order.')
    entries = sorted((k, v) for k, v in obs['repetition_counts'].items() if v > 0)
    if len(entries) > history_capacity:
        raise ValueError('Episode history exceeds this checkpoint capacity.')
    if not 0 <= obs['ply'] <= 1000 or not 2 <= obs['max_plies'] <= 1000:
        raise ValueError('PPO encoder supports episodes of at most 1000 plies.')
    history = np.zeros((history_capacity, 92), dtype=np.float32)
    for i, (key, count) in enumerate(entries):
        if not 1 <= count <= 3:
            raise ValueError('Invalid repetition count.')
        history[i, :91] = position_row(key)
        history[i, 91] = count / 3.0
    side = obs['side_to_move']
    key = ' '.join(obs['fen'].split()[:2])
    stats = np.array([
        side == 'red', (learner_side or side) == 'red',
        min(obs['halfmove_clock'], 120) / 120.0,
        obs['ply'] / 1000.0, obs['max_plies'] / 1000.0,
        obs['repetition_counts'].get(key, 0) / 3.0,
    ], dtype=np.float32)
    return {'board': np.asarray(obs['board_planes'], dtype=np.float32),
            'stats': stats, 'history': history}


def action_mask(obs):
    mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
    mask[np.asarray(obs['legal_action_ids'], dtype=np.int64)] = True
    return mask
