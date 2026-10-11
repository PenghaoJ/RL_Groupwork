import json
from pathlib import Path
import random
import re
import sys
import threading
import unittest
import urllib.request
import urllib.error
from unittest.mock import patch
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rules import Game, PRESETS, START_FEN, action_id, action_move, pseudo, in_check
from engines import UCIEngine, choose_move, close_workers
from training import XiangqiEnv
from app import Handler, LocalHTTPServer

class RulesTests(unittest.TestCase):
    def test_start_and_coordinates(self):
        game=Game()
        self.assertEqual(len(game.legal),44)
        for action in range(8100): self.assertEqual(action_id(action_move(action)),action)

    def test_piece_constraints(self):
        board=[[""]*9 for _ in range(10)]
        board[9][1]='H'; board[8][1]='P'
        self.assertFalse(pseudo(board,9,1,7,2)); board[8][1]=''
        self.assertTrue(pseudo(board,9,1,7,2))
        board[9][2]='E'; board[8][3]='P'
        self.assertFalse(pseudo(board,9,2,7,4)); board[8][3]=''
        self.assertTrue(pseudo(board,9,2,7,4))
        board[5][2]='E'; self.assertFalse(pseudo(board,5,2,3,4))
        board[7][7]='C'; board[2][7]='r'
        self.assertFalse(pseudo(board,7,7,2,7)); board[5][7]='P'
        self.assertTrue(pseudo(board,7,7,2,7)); board[4][7]='p'
        self.assertFalse(pseudo(board,7,7,2,7))
        board[6][0]='P'; self.assertFalse(pseudo(board,6,0,6,1)); self.assertFalse(pseudo(board,6,0,7,0))
        board[4][0]='P'; self.assertTrue(pseudo(board,4,0,4,1))

    def test_flying_general_and_pins(self):
        game=Game('4k4/9/9/9/4R4/9/9/9/9/4K4 b - - 0 1')
        self.assertTrue(in_check(game.board,'black'))
        # A blocker between the generals cannot leave their file.
        game=Game('4k4/9/9/9/4P4/9/9/9/9/4K4 w - - 0 1')
        self.assertNotIn('e5d5',game.legal)
        self.assertNotIn('e5f5',game.legal)
        self.assertIn('e5e6',game.legal)

    def test_bad_fens(self):
        for fen in ['bad','9/9/9/9/9/9/9/9/9/9 w','4k4/9/9/9/9/9/9/9/9/4K4 w','4k4/9/9/9/9/9/9/9/9/3K5 x']:
            with self.assertRaises(ValueError): Game(fen)

    def test_stalemate_is_loss(self):
        game=Game('4k4/3R1R3/9/9/9/9/9/9/9/3K5 b - - 0 1')
        self.assertEqual(game.outcome()['reason'],'stalemate')
        self.assertEqual(game.outcome()['winner'],'red')

    def test_checkmate(self):
        game=Game('4k4/4RR3/9/9/9/9/9/9/9/3K5 b - - 0 1')
        self.assertEqual(game.outcome()['reason'],'checkmate')

    def test_repetition_undo(self):
        game=Game()
        for move in ['b0c2','b9c7','c2b0','c7b9']*2: game.step(move)
        self.assertEqual(game.outcome()['reason'],'repetition')
        game.undo(); self.assertIsNone(game.outcome())
        while game.history: game.undo()
        self.assertEqual(game.fen,START_FEN)

    def test_truncation_and_env(self):
        env=XiangqiEnv(max_plies=2)
        obs,_=env.reset()
        self.assertEqual(len(obs['board_planes']),14)
        self.assertIn('repetition_counts',obs)
        env.step(action_id('a3a4'))
        _,reward,terminated,truncated,info=env.step('a6a5')
        self.assertEqual(reward,0); self.assertFalse(terminated); self.assertTrue(truncated)
        self.assertEqual(info['acting_side'],'black')

class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.engine=UCIEngine()
    @classmethod
    def tearDownClass(cls): cls.engine.close()

    def compare(self,game):
        lines=self.engine.perft(game.fen)
        expected={line.split(':')[0] for line in lines if re.match(r'^[a-i]\d[a-i]\d:',line)}
        self.assertEqual(set(game.legal),expected,game.fen)

    def test_legal_moves_match_pikafish(self):
        rng=random.Random(42)
        for preset in PRESETS:
            game=Game(preset['fen'])
            for _ in range(12):
                self.compare(game)
                if game.outcome(): break
                game.step(rng.choice(game.legal))

    def test_engine_and_custom_adapter(self):
        for model in ['pikafish','custom-example','random']:
            game=Game(PRESETS[0]['fen'])
            move,info=choose_move(model,game,100)
            self.assertIn(move,game.legal)

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=LocalHTTPServer(('127.0.0.1',0),Handler)
        cls.url=f'http://127.0.0.1:{cls.server.server_port}'
        threading.Thread(target=cls.server.serve_forever,daemon=True).start()
    @classmethod
    def tearDownClass(cls): cls.server.shutdown();cls.server.server_close()
    def call(self,path,data,origin=None):
        headers={'Content-Type':'application/json'}
        if origin:headers['Origin']=origin
        req=urllib.request.Request(self.url+'/api/'+path,json.dumps(data).encode(),headers)
        with urllib.request.urlopen(req) as r:return json.load(r)

    def test_episode_roundtrip_revision_isolation(self):
        one=self.call('session',{}); two=self.call('session',{})
        def req(path,data={}):return self.call(path,{'session':one['session'],'revision':one['state']['revision'],**data})
        state=req('move',{'move':one['state']['legal_moves'][0]})['state']
        with self.assertRaises(urllib.error.HTTPError) as e:req('undo')
        self.assertEqual(e.exception.code,409)
        one['state']=state
        episode=req('export')
        result=self.call('import',{'session':two['session'],'revision':0,'episode':episode})
        self.assertEqual(result['state']['fen'],state['fen'])
        bad={**episode,'moves':[{'move':'a0a9'}]}
        with self.assertRaises(urllib.error.HTTPError):self.call('import',{'session':two['session'],'revision':1,'episode':bad})
        exported=self.call('export',{'session':two['session'],'revision':1})
        self.assertEqual(exported['final_fen'],state['fen'])

    def test_cross_origin_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as e:self.call('session',{},'https://example.com')
        self.assertEqual(e.exception.code,403)

    def test_port_is_exclusive(self):
        with self.assertRaises(OSError): LocalHTTPServer(('127.0.0.1',self.server.server_port),Handler)

    def test_model_check_and_observation_are_read_only(self):
        one=self.call('session',{})
        args={'session':one['session'],'revision':0}
        check=self.call('models/check',{'model':'custom-team-template'})
        self.assertTrue(check['passed'])
        self.assertEqual({c['side'] for c in check['cases']},{'red','black'})
        observation=self.call('observation',args)
        self.assertEqual(observation['observation']['fen'],one['state']['fen'])
        self.assertEqual(observation['revision'],0)
        with patch('app.choose_move',side_effect=ValueError('Model returned an illegal move')):
            with self.assertRaises(urllib.error.HTTPError) as error:self.call('engine',{**args,'model':'custom-team-template'})
        self.assertEqual(error.exception.code,400)
        self.assertEqual(self.call('export',args)['moves'],[])
        close_workers()

    def test_english_only_without_locale_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('app.ROOT',Path(directory)):
                with urllib.request.urlopen(self.url+'/api/meta') as r:meta=json.load(r)
                self.assertEqual(meta['languages'],['en'])

if __name__=='__main__':unittest.main(verbosity=2)
