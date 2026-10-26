"""Check draw rewards and terminal targets without any learning updates."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from dqn_model import encode_state
from dqn_training import ENDGAMES, FixedOpponentEnv, legal_next_values
from rules import Game, START_FEN


class RewardDiagnostics(unittest.TestCase):
    def test_repetition_after_black_reply_is_terminal_zero(self):
        replies=iter(['b9c7','c7b9']*2)
        env=FixedOpponentEnv(opponent=lambda obs,rng: next(replies))
        env.reset(START_FEN)
        for index,move in enumerate(['b0c2','c2b0']*2):
            obs,reward,ended,limited,info=env.step(move)
            self.assertEqual(reward,0)
            self.assertFalse(limited)
            self.assertEqual(ended,index==3)
        self.assertEqual(info['outcome']['reason'],'repetition')
        self.assertEqual(env.game.repetitions[env.game.key],3)

    def test_repetition_after_red_move_stops_before_reply(self):
        game=Game(START_FEN)
        game.step('b0c2')
        replies=iter(['b9c7','c7b9']*2)
        env=FixedOpponentEnv(opponent=lambda obs,rng: next(replies))
        env.reset(game.fen)
        for index,move in enumerate(['c2b0','b0c2']*2):
            _,reward,ended,limited,info=env.step(move)
            self.assertEqual((reward,limited),(0,False))
            self.assertEqual(ended,index==3)
        self.assertEqual(info['outcome']['reason'],'repetition')
        self.assertEqual(info['environment_plies'],1)

    def test_capture_is_zero_and_resets_no_capture_clock(self):
        env=FixedOpponentEnv()
        env.reset(ENDGAMES['rook']['fen'],seed=42)
        _,reward,ended,limited,_=env.step('e2e5')
        self.assertEqual((reward,ended,limited),(0,False,False))
        self.assertEqual(env.game.history[0]['captured'],'p')
        self.assertEqual(env.game.halfmove,1)

    def test_no_capture_draw_is_terminal_zero(self):
        env=FixedOpponentEnv(opponent=lambda obs,rng: self.fail('Reply after terminal draw'))
        env.reset(START_FEN.rsplit(' ',2)[0]+' 119 1')
        _,reward,ended,limited,info=env.step('b0c2')
        self.assertEqual((reward,ended,limited),(0,True,False))
        self.assertEqual(info['outcome']['reason'],'no_capture')

    def test_draw_does_not_bootstrap_but_external_limit_does(self):
        class ConstantTarget(torch.nn.Module):
            def forward(self,batch):
                return torch.full((len(batch[0]),8100),7.0)
        state=encode_state(Game(START_FEN).observation())
        values=legal_next_values(ConstantTarget(),[state,state],[(),(1,)],
                                 [True,False],'cpu')
        self.assertEqual(values.tolist(),[0,7])
        self.assertFalse(values.requires_grad)

    def test_repetition_key_includes_side_and_excludes_clocks(self):
        game=Game(START_FEN)
        black=Game(START_FEN.replace(' w ',' b '))
        later=Game(START_FEN.rsplit(' ',2)[0]+' 12 7')
        self.assertNotEqual(game.key,black.key)
        self.assertEqual(game.key,later.key)


if __name__=='__main__':
    unittest.main(verbosity=2)
