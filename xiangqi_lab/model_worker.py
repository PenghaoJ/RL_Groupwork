"""Persistent Python policy host. stdout is reserved for the JSON protocol."""
from contextlib import redirect_stdout
import importlib.util
import json
from pathlib import Path
import sys
import traceback

def main():
    folder=Path(sys.argv[1]).resolve()
    manifest=json.loads((folder/'model.json').read_text(encoding='utf-8-sig'))
    entry=(folder/manifest.get('entrypoint','policy.py')).resolve()
    if not entry.is_relative_to(folder) or not entry.is_file():
        raise ValueError('The policy entrypoint must be a file inside the model folder.')
    checkpoint=manifest.get('checkpoint')
    if checkpoint:
        checkpoint=str((folder/checkpoint).resolve())
        if not Path(checkpoint).is_file(): raise FileNotFoundError('Checkpoint not found: '+checkpoint)
    sys.path.insert(0,str(folder))
    with redirect_stdout(sys.stderr):
        spec=importlib.util.spec_from_file_location('team_policy',entry)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        policy=module.Policy(checkpoint=checkpoint,device=manifest.get('device','cpu'))
    for line in sys.stdin:
        request=json.loads(line)
        try:
            if request.get('protocol') != 'xiangqi-lab.agent.v1':
                raise ValueError('Unsupported agent protocol.')
            with redirect_stdout(sys.stderr):
                result=policy.select_move(request['observation'],time_budget_ms=request['time_budget_ms'])
            if isinstance(result,str): result={'move':result}
            elif type(result) is int: result={'action_id':result}
            if not isinstance(result,dict): raise TypeError('Policy must return a move string, integer action ID or response dictionary.')
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            result={'error':str(exc)}
        result['request_id']=request['request_id']
        print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
