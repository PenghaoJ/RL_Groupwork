"""Use a Red-trained DQN on either side without changing its weights."""
from dqn_model import Policy as TrainedPolicy
from policy_perspective import as_red, flip_action


class Policy:
    def __init__(self, checkpoint=None, device="cpu"):
        self.policy = TrainedPolicy(checkpoint=checkpoint, device=device)

    def select_move(self, observation, time_budget_ms=800):
        normalized, flipped = as_red(observation)
        result = self.policy.select_move(normalized, time_budget_ms)
        action = result["action_id"]
        if flipped:
            action = flip_action(action)
        if action not in observation["legal_action_ids"]:
            raise ValueError("DQN returned an illegal action.")
        return {"action_id": action}
