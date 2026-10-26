"""Archive and inspect the received MCTS snapshots without running training."""
from pathlib import Path, PurePosixPath
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'output/mcts/RECEIVED_20261026'
SOURCE=Path('C:/wechat/临时文件/xwechat_files/wangpenghao1106_0003/msg/file/2026-10/mcts_ucb_v2删除后替换.zip')
sys.path.insert(0,str(ROOT/'xiangqi_lab'))
from rules import Game, PRESETS, parse_fen

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    DEST.mkdir(parents=True,exist_ok=False)
    shutil.copy2(SOURCE,DEST/'received_original.zip')
    manifest={'archive_sha256':sha(SOURCE),'source_path':str(SOURCE),'training':False,'versions':{}}
    with zipfile.ZipFile(SOURCE) as z:
        for info in z.infolist():
            parts=PurePosixPath(info.filename.replace('\\','/')).parts
            assert parts and not info.filename.startswith(('/','\\'))
            assert all(p not in ('.','..') and ':' not in p for p in parts)
            assert not ((info.external_attr>>16)&0o170000)==0o120000
            version=next((v for v in ('v2','v3','v4') if parts[0].startswith('mcts_ucb_'+v)),None)
            assert version,info.filename
            if info.is_dir() or '__pycache__' in parts or '.ipynb_checkpoints' in parts:continue
            target=DEST/version/Path(*parts[1:])
            assert target.resolve().is_relative_to(DEST.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(z.read(info))
    for version in ('v2','v3','v4'):
        folder=DEST/version
        data=json.loads((folder/'experience.json').read_text('utf-8'))
        assert data['preset_fens']==[p['fen'] for p in PRESETS[:4]]
        manifest['versions'][version]={'policy_sha256':sha(folder/'policy.py'),'experience_sha256':sha(folder/'experience.json'),
            'scenes':{key:{'games_in_memory':value['games'],'state_action_entries':len(value['values']),
                          'side_entries':dict(Counter(k.split('|')[0].split()[1] for k in value['values'])),
                          'values_sha256':hashlib.sha256(json.dumps(value['values'],sort_keys=True).encode()).hexdigest()}
                      for key,value in data['scenes'].items()}}
    assert len({v['policy_sha256'] for v in manifest['versions'].values()})==1
    policy_module=load(DEST/'v4/policy.py','received_mcts_preflight')
    checks=[]
    for scene,preset in enumerate(PRESETS[:3]):
        for side in ('red','black'):
            game=Game(preset['fen'])
            if side=='black':game.step(game.legal[0])
            obs=game.observation();unchanged=copy.deepcopy(obs)
            policy=policy_module.Policy();policy.rng.seed(2026102600+scene)
            # Explicit match context prevents missed scene selection when Black acts first at ply 1.
            policy.scene_id=scene
            started=time.perf_counter();move=policy.select_move(obs,800);elapsed=time.perf_counter()-started
            assert move in obs['legal_moves']
            assert obs==unchanged
            assert policy.scene_id==scene and policy.experience
            checks.append({'preset':preset['id'],'side':side,'move':move,'elapsed_ms':round(elapsed*1000,2),
                           'scene_id':policy.scene_id,'memory_entries':len(policy.experience)})
    # Diagnose the received rollout convention with a real one-ply mate, without changing it.
    mate=Game('4k4/3R1R3/9/9/9/9/9/9/9/3K5 w - - 0 1')
    probe=copy.deepcopy(mate);actual=probe.step('d8e8')
    assert actual['winner']=='red' and actual['reason']=='checkmate'
    node=policy_module.Node(copy.deepcopy(mate.board),mate.side,mate.halfmove,0,Counter(mate.repetitions),list(mate.legal))
    class ForcedMate:
        def choice(self,moves):
            assert 'd8e8' in moves
            return 'd8e8'
        def random(self):return 1.0
    received_value=policy_module._rollout(node,160,ForcedMate(),depth=10)
    manifest['preflight']={'legal_calls':checks,'observations_unchanged':True,'explicit_initial_scene_context':True,
        'rollout_probe':{'fen':mate.fen,'forced_move':'d8e8','actual_outcome':actual,
                         'expected_value_from_starting_side':1.0,'received_value':received_value,
                         'passed':received_value==1.0,
                         'handling':'Record the defect; retain original search code for the received-model match.'}}
    assert sha(DEST/'v4/experience.json')==manifest['versions']['v4']['experience_sha256']
    save(DEST/'inspection.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
