"""Frozen Double DQN vs the received v4 MCTS, with audited scene routing."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'xiangqi_lab'))
import torch
from rules import Game, PRESETS, action_move
from notebook_tools import replay_episode

RECEIVED=ROOT/'output/mcts/RECEIVED_20261026'
DDQN=ROOT/'xiangqi_lab/models/local_match_double_dqn'
DDQN_SHA='b616b5c8fdb10587a8b9054484348ee0db118b02cf2b4e6e5e54cba9d90d7379'
NAMES={'double_dqn':'Old Double DQN (best episode 200)','mcts_v4':'Received MCTS v4'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def tally(rows):
    c=Counter(r['result'] for r in rows)
    return {'games':len(rows),**{key:c[key] for key in ('double_dqn_win','mcts_v4_win','draw','ply_limit')}}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true',help='Resume only after verifying the saved plan and completed episodes.')
    args=parser.parse_args();out=args.output.resolve()
    if not args.resume:
        out.mkdir(parents=True,exist_ok=False);(out/'episodes').mkdir()
    else:
        assert (out/'plan.json').is_file() and (out/'progress.json').is_file()
    inspection=json.loads((RECEIVED/'inspection.json').read_text('utf-8'))
    bank_path=ROOT/'output/dqn/UNSEEN10_FOUR_20261025/boards.json'
    bank=json.loads(bank_path.read_text('utf-8-sig'))
    source={'policy':sha(RECEIVED/'v4/policy.py'),'experience':sha(RECEIVED/'v4/experience.json'),'ddqn':sha(DDQN/'best.pt')}
    assert source['ddqn']==DDQN_SHA
    assert source['policy']==inspection['versions']['v4']['policy_sha256']
    assert source['experience']==inspection['versions']['v4']['experience_sha256']
    boards=[{'suite':'existing10','board':b['id'],'group':b['group'],'fen':b['fen'],'scene':None} for b in bank['boards']]
    boards += [{'suite':'familiar3','board':'preset-'+p['id'],'group':p['id'],'fen':p['fen'],'scene':i} for i,p in enumerate(PRESETS[:3])]
    jobs=[]
    for i,b in enumerate(boards):
        for red,black in [('double_dqn','mcts_v4'),('mcts_v4','double_dqn')]:
            jobs.append({**b,'red':red,'black':black,'seed':2026102601+i,'id':f"{b['suite']}_{b['board']}_red_{red}"})
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'training':False,'jobs':jobs,'bank_sha256':sha(bank_path),
          'rules_sha256':sha(ROOT/'xiangqi_lab/rules.py'),'source_hashes':source,
          'max_plies':160,'requested_move_budget_ms':800,'mcts_max_iterations':96,'mcts_rollout_depth':10,
          'mcts_timing':'Received code uses an 80% deadline reserve and a four-simulation minimum. Actual time and completed simulations are recorded.',
          'randomness':'One fixed MCTS seed per board, reused across player assignments. Wall-clock search stopping means seeds do not guarantee identical reruns.',
          'scene_routing':'Fresh MCTS instance per game; preset scene supplied explicitly for either side. Custom starts use received unknown-start search behavior.',
          'black_play':'Double DQN uses its existing rotate/color-swap adapter. MCTS uses its native player-to-move search; stored experience contains Red moves only.',
          'model_selection':'User selected latest v4 before testing. No match-based selection or parameter tuning.',
          'known_received_defect':inspection['preflight']['rollout_probe'],
          'limits':['Exploratory 26-game snapshot, not an algorithm-wide ranking.','Familiar preset results are separate from the existing ten-board regression set.','Search budget is not matched training or inference compute.','Received rollout sign defect is retained, not silently repaired.']}
    if args.resume:
        previous=json.loads((out/'plan.json').read_text('utf-8'))
        for key in ('jobs','source_hashes','bank_sha256','rules_sha256','max_plies','requested_move_budget_ms','mcts_max_iterations','mcts_rollout_depth'):
            assert previous[key]==plan[key],key
        plan=previous
    else:
        save(out/'plan.json',plan)
        save(out/'inspection.json',inspection)
    mcts=load(RECEIVED/'v4/policy.py','mcts_received_match')
    ddqn=load(DDQN/'policy.py','ddqn_selected_match').Policy(str(DDQN/'best.pt'),device='cpu')
    net=ddqn.policy.model;net.eval();net.requires_grad_(False)
    before={n:t.detach().clone() for n,t in net.state_dict().items()}
    # Read-only search instrumentation: preserve the received code's return values.
    counts={'simulations':0,'prior_lookups':0,'prior_hits':0}
    raw_rollout=mcts._rollout;raw_combined=mcts._combined_value
    def counted_rollout(*a,**k):
        counts['simulations']+=1
        return raw_rollout(*a,**k)
    def counted_combined(parent,child):
        counts['prior_lookups']+=1
        if parent.experience.get(mcts._experience_key(parent.board,parent.side,child.move),{}).get('visits',0)>0:
            counts['prior_hits']+=1
        return raw_combined(parent,child)
    mcts._rollout=counted_rollout;mcts._combined_value=counted_combined
    rows=[];latencies={k:[] for k in NAMES};all_simulations=[];start=time.perf_counter()
    if args.resume:
        progress=json.loads((out/'progress.json').read_text('utf-8'))
        rows=progress['rows']
        assert len(rows)==progress['completed']
        for expected,row in zip(jobs,rows):
            assert row['replay']==f"episodes/{expected['id']}.json"
            ep=json.loads((out/row['replay']).read_text('utf-8'))
            replayed=replay_episode(ep)
            assert ep['match']['id']==expected['id'] and replayed.outcome()==ep['outcome']
            for move in ep['moves']:
                actor=next(k for k,v in NAMES.items() if v==move['actor'])
                latencies[actor].append(move['analysis']['elapsed_ms'])
                if actor=='mcts_v4':all_simulations.append(move['analysis']['simulations'])
        save(out/'resume_audit.json',{'resumed_utc':datetime.now(timezone.utc).isoformat(),'completed_games_preserved':len(rows),
              'cause':'The audit compared an empty, not-yet-loaded table with its first lazy load after a forced move.',
              'fix':'Only establish the loaded-memory hash once _load_experience has populated its cache marker.',
              'discarded_partial_game':'familiar3_preset-rook_red_double_dqn','model_code_or_search_settings_changed':False,
              'policy_and_experience_hashes_verified':True})
    completed_before=len(rows)
    try:
        with torch.inference_mode():
            for number,job in enumerate(jobs,1):
                if number<=completed_before:continue
                policy=mcts.Policy();policy.rng.seed(job['seed']);policy.scene_id=job['scene']
                assert policy.iterations==96 and policy.rollout_depth==10
                initial_memory_hash=None
                game=Game(job['fen'],max_plies=160);assert game.outcome() is None
                prior_hits=0;simulations=0;mcts_moves=0;maximum_entries=0
                last_print=time.monotonic()
                while game.outcome() is None:
                    actor=job[game.side];observation=game.observation();unchanged=copy.deepcopy(observation)
                    counts.update(simulations=0,prior_lookups=0,prior_hits=0)
                    t=time.perf_counter()
                    result=policy.select_move(observation,800) if actor=='mcts_v4' else ddqn.select_move(observation,800)
                    elapsed_ms=(time.perf_counter()-t)*1000
                    assert observation==unchanged
                    if isinstance(result,str):move=result
                    elif isinstance(result,dict):move=result['move'] if 'move' in result else action_move(result['action_id'])
                    else:move=action_move(result)
                    assert move in game.legal,(job['id'],move)
                    analysis={'elapsed_ms':round(elapsed_ms,3)}
                    latencies[actor].append(elapsed_ms)
                    if actor=='mcts_v4':
                        mcts_moves+=1;simulations+=counts['simulations'];prior_hits+=counts['prior_hits']
                        maximum_entries=max(maximum_entries,len(policy.experience))
                        all_simulations.append(counts['simulations'])
                        analysis.update(counts,scene_id=policy.scene_id,memory_entries=len(policy.experience),seed=job['seed'])
                        if policy._experience_mtime is not None:
                            memory_hash=hashlib.sha256(json.dumps(policy.experience,sort_keys=True).encode()).hexdigest()
                            if initial_memory_hash is None:initial_memory_hash=memory_hash
                            assert memory_hash==initial_memory_hash,'Loaded memory changed during inference'
                    game.step(move,actor=NAMES[actor],analysis=analysis)
                    if time.monotonic()-last_print>25:
                        print(f"{number:02}/{len(jobs)} {job['board']} Red={job['red']}: {len(game.history)} plies in progress",flush=True)
                        last_print=time.monotonic()
                outcome=game.outcome();episode=game.export()
                replayed=replay_episode(episode)
                assert replayed.fen==game.fen and replayed.outcome()==outcome
                episode['match']={**job,'training':False,'budget_ms':800}
                replay=f"episodes/{job['id']}.json";save(out/replay,episode)
                winner=job[outcome['winner']] if outcome['winner'] else None
                result=winner+'_win' if winner else 'ply_limit' if outcome['truncated'] else 'draw'
                row={'suite':job['suite'],'board':job['board'],'group':job['group'],'red':job['red'],'black':job['black'],
                     'result':result,'winner_side':outcome['winner'],'reason':outcome['reason'],'plies':len(game.history),
                     'seed':job['seed'],'mcts_moves':mcts_moves,'mcts_simulations':simulations,'mcts_prior_hits':prior_hits,
                     'max_memory_entries':maximum_entries,'replay':replay,'final_fen':game.fen}
                rows.append(row);save(out/'progress.json',{'completed':len(rows),'planned':len(jobs),'totals':tally(rows),'rows':rows})
                print(f"{number:02}/{len(jobs)} {job['board']} Red={job['red']}: {result}, {outcome['reason']}, {len(game.history)} plies; memory hits={prior_hits}",flush=True)
    except Exception as e:
        save(out/'failure.json',{'error':repr(e),'completed':len(rows),'job':job,'last_episode':game.export()})
        raise
    assert sha(DDQN/'best.pt')==source['ddqn']
    assert sha(RECEIVED/'v4/policy.py')==source['policy']
    assert sha(RECEIVED/'v4/experience.json')==source['experience']
    assert all(torch.equal(t,net.state_dict()[n]) for n,t in before.items())
    summary={'status':'complete','completed_utc':datetime.now(timezone.utc).isoformat(),'training':False,
             'ddqn_weights_unchanged':True,'mcts_source_and_experience_unchanged':True,'loaded_memory_unchanged':True,
             'observations_unchanged':True,'all_episodes_replayed':True,'source_hashes':source,
             'totals':tally(rows),'suites':{s:tally([r for r in rows if r['suite']==s]) for s in ('existing10','familiar3')},
             'by_group':{s:{g:tally([r for r in rows if r['suite']==s and r['group']==g]) for g in ('rook','cannon','horse')} for s in ('existing10','familiar3')},
             'by_red':{s:{actor:tally([r for r in rows if r['suite']==s and r['red']==actor]) for actor in NAMES} for s in ('existing10','familiar3')},
             'reasons':dict(Counter(r['reason'] for r in rows)),'total_plies':sum(r['plies'] for r in rows),
             'inference':{k:{'moves':len(v),'mean_ms':round(statistics.mean(v),3),'max_ms':round(max(v),3)} for k,v in latencies.items()},
             'mcts_simulations':{'total':sum(all_simulations),'mean_per_move':statistics.mean(all_simulations),'maximum':max(all_simulations)},
             'total_prior_hits':sum(r['mcts_prior_hits'] for r in rows),
             'elapsed_since_plan_seconds':round((datetime.now(timezone.utc)-datetime.fromisoformat(plan['created_utc'])).total_seconds(),2),
             'resumed_completed_games':completed_before,'runner_sha256':sha(Path(__file__)),
             'torch_version':torch.__version__,'known_received_defect':plan['known_received_defect'],'bank_sha256':plan['bank_sha256']}
    save(out/'summary.json',summary)
    with (out/'games.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
