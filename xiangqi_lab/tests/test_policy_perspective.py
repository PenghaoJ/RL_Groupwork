"""Check color symmetry, legal moves, counters and terminal outcomes."""
import copy
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_perspective import as_red, flip_action, flip_fen, flip_move, flip_observation
from rules import Game, PRESETS, action_id


class PerspectiveTests(unittest.TestCase):
    def test_all_action_ids_round_trip(self):
        for action in range(8100):
            self.assertEqual(flip_action(flip_action(action)), action)

    def test_games_preserve_rules_and_terminal_outcomes(self):
        rng = random.Random(42)
        for preset in PRESETS:
            original = Game(preset["fen"], max_plies=80)
            flipped = Game(flip_fen(preset["fen"]), max_plies=80)
            for _ in range(81):
                obs = original.observation()
                saved = copy.deepcopy(obs)
                transformed = flip_observation(obs)
                self.assertEqual(flip_observation(transformed), obs)
                self.assertEqual(obs, saved)
                self.assertEqual(set(transformed["legal_moves"]), set(flipped.legal))
                self.assertEqual(transformed["repetition_counts"], dict(flipped.repetitions))
                self.assertEqual(transformed["board_planes"], flipped.observation()["board_planes"])
                self.assertEqual(transformed["ply"], len(flipped.history))
                self.assertEqual(transformed["halfmove_clock"], flipped.halfmove)
                normalized, changed = as_red(obs)
                self.assertEqual(normalized["side_to_move"], "red")
                self.assertEqual(changed, obs["side_to_move"] == "black")
                outcome = original.outcome()
                if outcome:
                    expected = dict(outcome)
                    if outcome["winner"]:
                        expected["winner"] = "black" if outcome["winner"] == "red" else "red"
                    self.assertEqual(flipped.outcome(), expected)
                    break
                move = rng.choice(original.legal)
                self.assertEqual(flip_action(action_id(move)), action_id(flip_move(move)))
                original.step(move)
                flipped.step(flip_move(move))


if __name__ == "__main__":
    unittest.main()
