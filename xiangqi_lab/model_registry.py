"""Discover local model manifests without executing model code."""
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parent

def models():
    items=[]
    config=ROOT/'models.json'
    legacy=json.loads(config.read_text(encoding='utf-8-sig')) if config.exists() else []
    if not isinstance(legacy,list): raise ValueError('models.json must contain a list.')
    items.extend({**m,'_folder':str(ROOT)} for m in legacy if isinstance(m,dict))
    if len(items)!=len(legacy): raise ValueError('Each model definition must be an object.')
    for file in sorted((ROOT/'models').glob('*/model.json')):
        manifest=json.loads(file.read_text(encoding='utf-8-sig'))
        if not isinstance(manifest,dict) or manifest.get('schema')!='xiangqi-lab.model.v1':
            raise ValueError('Unsupported model manifest: '+str(file))
        items.append({**manifest,'_folder':str(file.parent.resolve())})
    ids=set()
    for item in items:
        mid=item.get('id','')
        if not isinstance(mid,str) or not re.fullmatch(r'custom-[a-z0-9_-]+',mid): raise ValueError('Model IDs must start with custom- and use lowercase letters, digits, hyphens or underscores.')
        if mid in ids: raise ValueError('Duplicate model ID: '+mid)
        ids.add(mid)
    for item in items:
        mid=item['id']
        if not isinstance(item.get('name'),str) or not item['name'].strip(): raise ValueError('Model name is required: '+mid)
        if item.get('protocol') not in ('python','json','uci'): raise ValueError('Model protocol must be python, json or uci.')
        timeout=item.get('timeout_seconds',30)
        if type(timeout) not in (int,float) or not .1<=timeout<=120: raise ValueError('Model timeout_seconds must be between 0.1 and 120.')
        if item['protocol']=='python':
            folder=Path(item['_folder'])
            entry=item.get('entrypoint','policy.py')
            if not isinstance(entry,str): raise ValueError('Policy entrypoint must be a path string.')
            path=(folder/entry).resolve()
            if not path.is_relative_to(folder) or not path.is_file(): raise ValueError('Policy entrypoint not found: '+mid)
            if not isinstance(item.get('python','{python}'),str): raise ValueError('Model python must be an executable path.')
        elif not isinstance(item.get('command'),list) or not item['command'] or not all(isinstance(x,str) for x in item['command']):
            raise ValueError('Model command must be a non-empty string array.')
    return items

def command_for(config):
    def expand(arg):
        return arg.replace('{python}',sys.executable).replace('{root}',str(ROOT)).replace('{model_dir}',config['_folder'])
    if config['protocol']=='python':
        return [expand(config.get('python','{python}')),'-u',str(ROOT/'model_worker.py'),config['_folder']]
    return [expand(arg) for arg in config['command']]

def catalog():
    builtins=[{'id':'human','name':'Human','description':'Click the board to move.'},{'id':'pikafish','name':'Pikafish','description':'Bundled NNUE reference engine.'},{'id':'random','name':'Random legal policy','description':'Untrained baseline.'}]
    return builtins+[{'id':m['id'],'name':m['name'],'description':m.get('description','Local policy'),'protocol':m['protocol']} for m in models()]
