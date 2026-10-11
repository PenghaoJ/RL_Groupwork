"""Runnable integration template, NOT a trained model.

Copy this folder, change model.json's id/name, then implement this class.
Load your network/checkpoint once in __init__. Select a legal action in
select_move. Do not keep game state here: one worker serves multiple games.
"""
class Policy:
    def __init__(self, checkpoint=None, device='cpu'):
        self.checkpoint=checkpoint
        self.device=device
        # Load your architecture and weights here, then switch to eval mode.
        # Use your own Python environment in model.json if you need torch.

    def select_move(self, observation, time_budget_ms=800):
        # Input planes: [14, 10, 9], fixed Red/Black orientation.
        # Use observation['legal_action_ids'] to mask a policy's 8100 logits.
        # This deterministic baseline exists only to test the interface.
        return observation['legal_moves'][0]
