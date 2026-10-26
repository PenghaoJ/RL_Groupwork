"""Audit rewards, repetition records and frozen policies without training."""
from collections import Counter
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xiangqi_lab"))
import torch
from dqn_model import batch_states, encode_state
from rules import Game, action_id
from compare_four_models import load_policy, sha, write_json

TEST = ROOT / "output/dqn/UNSEEN10_FOUR_20261025"
OUT = ROOT / "output/dqn/DIAGNOSIS_ROOK20_20261025"


def key(fen):
    return " ".join(fen.split()[:2])


def summary(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else None


def restore(episode, index):
    """Recover a recorded decision state with its actual repetition history."""
    game = Game(episode["moves"][index]["fen_before"], episode["max_plies"])
    game.initial_fen = episode["initial_fen"]
    game.history = copy.deepcopy(episode["moves"][:index])
    game.repetitions = Counter([key(episode["initial_fen"])]+[key(m["fen_after"]) for m in episode["moves"][:index]])
    return game


def branch(game, move):
    child = copy.copy(game)
    child.board = copy.deepcopy(game.board)
    child.history = list(game.history)
    child.repetitions = game.repetitions.copy()
    child._legal = None
    child.step(move)
    return child


def training_audit(package):
    with zipfile.ZipFile(package) as z:
        prefix = next(n for n in z.namelist() if n.endswith('/training_report/episodes.csv')).rsplit('/',1)[0]+'/'
        rows = list(csv.DictReader(io.StringIO(z.read(prefix+'episodes.csv').decode())))
        config = json.loads(z.read(prefix+'config.json'))
        result = json.loads(z.read(prefix+'summary.json'))
        validation = list(csv.DictReader(io.StringIO(z.read(prefix+'validation.csv').decode())))
    total_steps = int(rows[-1]['learner_steps'])
    nonzero = [r for r in rows if float(r['reward']) != 0]
    last_replay = [r for r in nonzero if int(r['learner_steps']) > total_steps-config['replay_size']]
    best_rows = rows[:result['best_episode']]
    def block(selected):
        return dict(episodes=len(selected), wins=sum(float(r['reward'])>0 for r in selected),
            losses=sum(float(r['reward'])<0 for r in selected),
            reasons=dict(Counter(r['reason'] for r in selected)),
            epsilon_at_end=float(selected[-1]['epsilon']),
            total_steps_at_end=int(selected[-1]['learner_steps']),
            logged_last_batch_loss=summary([float(r['loss']) for r in selected if r['loss']]))
    return dict(episodes=len(rows), total_steps=total_steps, nonzero_reward_transitions=len(nonzero),
        nonzero_reward_fraction=len(nonzero)/total_steps,
        nonzero_rewards_in_final_replay=len(last_replay), final_replay_size=min(total_steps,config['replay_size']),
        expected_nonzero_rewards_per_uniform_batch=len(last_replay)/min(total_steps,config['replay_size'])*config['batch_size'],
        best_episode=result['best_episode'], best_checkpoint_experience=block(best_rows),
        starts_seen_at_best=dict(Counter(r.get('board_id','single-rook-start') for r in best_rows)),
        positive_episodes_by_start_at_best=dict(Counter(r.get('board_id','single-rook-start') for r in best_rows if float(r['reward'])>0)),
        last_100=block(rows[-100:]), first_100=block(rows[:100]),
        blocks=[dict(start=lo,end=hi,**block(rows[lo-1:hi])) for lo,hi in [(1,100),(101,500),(501,1000),(1001,1500)]],
        validation=validation,
        limitations='CSV loss is the last sampled minibatch loss per episode, not an episode or full-run mean.')


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    plan = json.loads((TEST/'plan.json').read_text())
    bank = json.loads((TEST/'boards.json').read_text())
    report = dict(training=False, optimizer_steps=0, models={}, checks={},
                  source_hashes={n:sha(ROOT/'xiangqi_lab'/n) for n in ('rules.py','dqn_model.py','dqn_training.py')})
    for name, metadata in plan['models'].items():
        data = training_audit(metadata['original_path'])
        episodes = [(p,json.loads(p.read_text())) for p in sorted((TEST/'results/episodes').glob(name+'_*.json'))]
        assert len(episodes)==200
        repetition, lengths, red_terminal, red_backtracks, red_steps = [], [], 0, 0, 0
        total_decisions, full_returns, positive_returns = 0, [], []
        for path, episode in episodes:
            moves = episode['moves']
            counts = Counter({key(episode['initial_fen']):1})
            first_third = None
            prior_red = None
            for ply,m in enumerate(moves,1):
                counts[key(m['fen_after'])] += 1
                if counts[key(m['fen_after'])] >= 3 and first_third is None:
                    first_third = ply
                if m['side']=='red':
                    red_steps += 1
                    red_backtracks += int(prior_red is not None and m['move']==prior_red[2:]+prior_red[:2])
                    prior_red = m['move']
            outcome = episode['outcome']
            assert first_third is None or first_third == len(moves)
            if outcome['reason']=='repetition':
                assert counts[key(episode['final_fen'])] == 3 and first_third == len(moves)
                repetition.append((path,episode))
                lengths.append(len(moves))
                red_terminal += moves[-1]['side']=='red'
        data['test_repetition'] = dict(games=len(repetition), fraction=len(repetition)/200,
            red_made_final_move=red_terminal, black_made_final_move=len(repetition)-red_terminal,
            length_plies=summary(lengths), final_positions_seen_exactly_three_times=True,
            red_immediate_reverse_moves=red_backtracks, red_moves=red_steps,
            red_reverse_move_fraction=red_backtracks/red_steps,
            note='A reverse move may be legal and useful; this count alone does not prove a mistake.')
        model,saved = load_policy(metadata['original_path'],metadata)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        start_q=[]
        for b in bank['boards']:
            g=Game(b['fen'])
            values=model(batch_states([encode_state(g.observation())],'cpu'))[0]
            legal=[action_id(m) for m in g.legal]
            start_q.append(dict(board=b['id'],max_legal_q=float(values[legal].max()),
                                min_legal_q=float(values[legal].min())))
        data['start_values']=start_q
        # Inspect at most three Red-ended repetition cases per rook board.
        chosen=[]
        for board in [b for b in bank['boards'] if b['group']=='rook']:
            selected=[(p,e) for p,e in repetition if e['evaluation']['board']==board['id'] and e['moves'][-1]['side']=='red'][:3]
            chosen.extend(selected)
        probes=[]
        for path,episode in chosen:
            i=len(episode['moves'])-1
            game=restore(episode,i)
            obs=game.observation()
            values=model(batch_states([encode_state(obs)],'cpu'))[0]
            actual=episode['moves'][i]['move']
            greedy=max(game.legal,key=lambda m:float(values[action_id(m)]))
            assert greedy==actual
            branches=[]
            for move in game.legal:
                child=branch(game,move)
                outcome=child.outcome()
                branches.append(dict(move=move,q=float(values[action_id(move)]),outcome=outcome))
            ongoing=[b for b in branches if b['outcome'] is None]
            wins=[b for b in branches if b['outcome'] and b['outcome']['winner']=='red']
            # Change only the next repeated position's past count in a diagnostic observation.
            altered=copy.deepcopy(obs)
            repeated_key=key(episode['final_fen'])
            assert altered['repetition_counts'][repeated_key]==2
            altered['repetition_counts'][repeated_key]=1
            q_without_third=model(batch_states([encode_state(altered)],'cpu'))[0]
            probes.append(dict(episode_file=str(path.relative_to(ROOT)),board=episode['evaluation']['board'],
                red_ply=i+1,move=actual,q=float(values[action_id(actual)]),true_immediate_return=0,
                q_if_next_position_was_seen_once=float(q_without_third[action_id(actual)]),
                legal_moves=len(branches),continuing_alternatives=len(ongoing),immediate_winning_alternatives=wins,
                top_alternatives=sorted(branches,key=lambda b:-b['q'])[:5],fen_before=game.fen))
        data['terminal_repetition_q_probes']=probes
        # Look for one-move wins in two preselected rook games per board.
        tactical=[]
        for board in [b for b in bank['boards'] if b['group']=='rook']:
            for repeat in (1,2):
                path=TEST/'results/episodes'/f"{name}_{board['id']}_{repeat:02d}.json"
                episode=json.loads(path.read_text())
                first_red_decisions=[i for i,e in enumerate(episode['moves']) if e['side']=='red'][:8]
                for i in first_red_decisions:
                    entry=episode['moves'][i]
                    game=restore(episode,i)
                    winning=[]
                    for move in game.legal:
                        # A rook win requires a rook move or pawn/king support; inspect all legal moves.
                        out=branch(game,move).outcome()
                        if out and out['winner']=='red': winning.append(move)
                    if winning:
                        values=model(batch_states([encode_state(game.observation())],'cpu'))[0]
                        tactical.append(dict(board=board['id'],repeat=repeat,red_ply=i+1,
                            selected=entry['move'],selected_wins=entry['move'] in winning,
                            winning_moves=winning,selected_q=float(values[action_id(entry['move'])]),
                            win_q={m:float(values[action_id(m)]) for m in winning},
                            episode_file=str(path.relative_to(ROOT)),fen_before=game.fen))
            print(name,board['id'],'diagnostic move checks complete',flush=True)
        data['tactical_probe'] = dict(sample='First eight Red decisions in rook boards 1-4, original repeats 1 and 2: 8 games per model. Diagnostic sample, not a new benchmark.',
            opportunities=len(tactical),missed=sum(not t['selected_wins'] for t in tactical),details=tactical)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        assert sha(metadata['original_path'])==metadata['file_sha256']
        data['weights_unchanged']=True
        report['models'][name]=data
        write_json(OUT/'diagnosis.json',report)
        print(name, 'nonzero rewards',data['nonzero_reward_transitions'],'/',data['total_steps'],
              'repetitions',len(repetition),'terminal probes',len(probes),
              'missed one-move wins',data['tactical_probe']['missed'],'/',len(tactical),flush=True)
    report['checks']=dict(all_800_repetition_counters_verified=True,all_weights_unchanged=True,
                           no_training=True,production_sources_unchanged=all(sha(ROOT/'xiangqi_lab'/n)==h for n,h in report['source_hashes'].items()))
    write_json(OUT/'diagnosis.json',report)
    print('Diagnostic audit complete. No training.',flush=True)


if __name__=='__main__':
    main()
