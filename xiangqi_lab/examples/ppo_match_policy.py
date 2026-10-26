"""Load the delivered PPO actor safely and use Red's view on either side."""
import io
import json
from pathlib import Path
import zipfile

from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
import torch

from ppo_encoding import ENCODING_VERSION, action_mask, encode_observation, observation_space
from ppo_features import HistoryFeatures
from policy_perspective import as_red, flip_action


class Policy:
    def __init__(self, checkpoint=None, device="cpu"):
        if checkpoint is None:
            raise ValueError("PPO checkpoint is required.")
        folder = Path(checkpoint).resolve().parent
        self.encoder = json.loads((folder / "encoder.json").read_text(encoding="utf-8"))
        if self.encoder["version"] != ENCODING_VERSION or self.encoder["trained_side"] != "red":
            raise ValueError("Expected the delivered Red-trained PPO encoder.")
        torch.set_num_threads(1)
        with zipfile.ZipFile(checkpoint) as archive:
            data = json.loads(archive.read("data"))
            kwargs = data["policy_kwargs"]
            if kwargs.get("net_arch") != {"pi": [64], "vf": [64]} or kwargs.get("features_extractor_class") != "<class 'ppo_features.HistoryFeatures'>":
                raise ValueError("Unsupported PPO network architecture.")
            # Load tensors only; serialized Python objects and optimizer state are unused.
            state = torch.load(io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
        self.model = MaskableMultiInputActorCriticPolicy(
            observation_space(self.encoder["history_capacity"]), spaces.Discrete(8100),
            lr_schedule=lambda _: 0.0, features_extractor_class=HistoryFeatures,
            net_arch={"pi": [64], "vf": [64]}, activation_fn=torch.nn.Tanh,
        ).to(device)
        self.model.load_state_dict(state, strict=True)
        self.model.set_training_mode(False)
        self.model.requires_grad_(False)

    def select_move(self, observation, time_budget_ms=800):
        normalized, flipped = as_red(observation)
        mask = action_mask(normalized)
        if not mask.any():
            raise ValueError("No legal moves in a terminal position.")
        encoded = encode_observation(normalized, self.encoder["history_capacity"])
        with torch.inference_mode():
            action, _ = self.model.predict(encoded, deterministic=True, action_masks=mask)
        action = int(action)
        if flipped:
            action = flip_action(action)
        if action not in observation["legal_action_ids"]:
            raise ValueError("PPO returned an illegal action.")
        return action
