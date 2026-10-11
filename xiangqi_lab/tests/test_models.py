import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import engines
import model_registry
from policy_process import close_workers, POOL
from rules import Game, PRESETS

class ModelContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.folder=Path(self.temp.name)
        self.config={'schema':'xiangqi-lab.model.v1','id':'custom-test','name':'Test policy','protocol':'python','entrypoint':'policy.py','python':'{python}','checkpoint':None,'device':'cpu','timeout_seconds':3,'_folder':str(self.folder)}
    def tearDown(self):close_workers();self.temp.cleanup()
    def policy(self,source,**config):
        self.config.update(config)
        (self.folder/'policy.py').write_text(source,encoding='utf-8')
        (self.folder/'model.json').write_text(json.dumps(self.config),encoding='utf-8')
    def choose(self,game=None):
        with patch('engines.custom_models',return_value=[self.config]):
            return engines.choose_move('custom-test',game or Game(PRESETS[0]['fen']),100)

    def test_persistent_loading_and_action_ids(self):
        self.policy('''from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
@dataclass
class Options:
    device: str = 'cpu'
class Policy:
    def __init__(self,checkpoint=None,device='cpu'):
        self.options=Options(device)
        with Path('loads.txt').open('a') as f:f.write('loaded\\n')
        print('A model library may print startup diagnostics')
    def select_move(self,observation,time_budget_ms=800):
        print('Inference diagnostic')
        return observation['legal_action_ids'][0]
''')
        game=Game(PRESETS[0]['fen'])
        for _ in range(2):
            move,_=self.choose(game);self.assertIn(move,game.legal);game.step(move)
        self.assertEqual((self.folder/'loads.txt').read_text(),'loaded\n')
        self.assertEqual(len(POOL),1)

    def test_illegal_action_does_not_change_game(self):
        self.policy("class Policy:\n def __init__(self,**kwargs):pass\n def select_move(self,*args,**kwargs):return 'a0a9'\n")
        game=Game(PRESETS[0]['fen']);before=game.export()
        with self.assertRaisesRegex(ValueError,'illegal move'):self.choose(game)
        self.assertEqual(game.export(),before)

    def test_timeout_kills_worker(self):
        self.policy("import time\nclass Policy:\n def __init__(self,**kwargs):pass\n def select_move(self,*args,**kwargs):time.sleep(2)\n",timeout_seconds=.2)
        with self.assertRaisesRegex(RuntimeError,'timed out'):self.choose()
        self.assertTrue(all(w.process.poll() is not None for w in POOL.values()))

    def test_missing_checkpoint_and_bad_environment(self):
        self.policy("class Policy:\n def __init__(self,**kwargs):pass\n",checkpoint='missing.pt')
        with self.assertRaisesRegex(RuntimeError,'Checkpoint not found'):self.choose()
        close_workers();self.config['python']=str(self.folder/'missing-python')
        with self.assertRaises(FileNotFoundError):self.choose()

    def test_inference_exception_is_reported(self):
        self.policy("class Policy:\n def __init__(self,**kwargs):pass\n def select_move(self,*args,**kwargs):raise RuntimeError('shape mismatch')\n")
        with self.assertRaisesRegex(RuntimeError,'shape mismatch'):self.choose()

    def test_malformed_json_agent(self):
        self.config.update(protocol='json',command=[sys.executable,'-c',"print('not-json')"])
        with self.assertRaisesRegex(ValueError,'exactly one JSON'):self.choose()

    def test_ambiguous_response_is_rejected(self):
        self.policy("class Policy:\n def __init__(self,**kwargs):pass\n def select_move(self,o,**kwargs):return {'move':o['legal_moves'][0],'action_id':o['legal_action_ids'][0]}\n")
        with self.assertRaisesRegex(ValueError,'exactly one'):self.choose()

    def test_custom_uci_agent(self):
        self.config.update(protocol='uci',command=[str(engines.engine_path())],_folder=str(engines.engine_path().parent))
        game=Game(PRESETS[0]['fen']);move,_=self.choose(game)
        self.assertIn(move,game.legal)

    def test_folder_discovery_and_duplicate_ids(self):
        folder=self.folder/'models'/'my_policy';folder.mkdir(parents=True)
        (folder/'policy.py').write_text('class Policy: pass')
        manifest={k:v for k,v in self.config.items() if not k.startswith('_')}
        (folder/'model.json').write_text(json.dumps(manifest))
        with patch.object(model_registry,'ROOT',self.folder):
            self.assertEqual(model_registry.models()[0]['id'],'custom-test')
            (self.folder/'models.json').write_text(json.dumps([manifest]))
            with self.assertRaisesRegex(ValueError,'Duplicate'):model_registry.models()

if __name__=='__main__':unittest.main()
