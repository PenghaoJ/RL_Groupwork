"""Write the frozen-policy diagnosis and a disabled follow-up experiment plan."""
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

from diagnose_rook_models import OUT, ROOT, restore, branch
sys.path.insert(0,str(ROOT/'xiangqi_lab'))
from notebook_tools import board_svg
from compare_four_models import write_json, sha


def main():
    data=json.loads((OUT/'diagnosis.json').read_text())
    engine_checks=json.loads((OUT/'pikafish-verification.json').read_text())
    models=data['models']
    cases=[]
    (OUT/'examples').mkdir(exist_ok=True)
    for checked in engine_checks:
        name=checked['model']
        probe=next(p for p in models[name]['terminal_repetition_q_probes'] if p['episode_file']==checked['episode_file'])
        original=ROOT/probe['episode_file']
        episode=json.loads(original.read_text())
        game=restore(episode,len(episode['moves'])-1)
        winning=next(w for w in probe['immediate_winning_alternatives'] if w['move']==checked['winning_move'])
        alternative=branch(game,winning['move']).export()
        alternative['diagnostic']={'counterfactual':True,'model_did_not_play_this_move':True,
                                    'purpose':'Verify an available one-move win; no training data is created.'}
        actual_name=f'{name}_actual_draw.json'
        alternative_name=f'{name}_alternative_not_model_play.json'
        shutil.copy2(original,OUT/'examples'/actual_name)
        write_json(OUT/'examples'/alternative_name,alternative)
        svg=board_svg(game,'Before the decision')
        ET.fromstring(svg)
        cases.append(dict(name=name,probe=probe,win=winning,svg=svg,
                          actual=actual_name,alternative=alternative_name))
    proposal={
        'status':'proposal_only','training_enabled':False,'production_code_changed':False,
        'preserve':['all four best models','existing research draw rules','original win/loss evaluation reward','existing test reports'],
        'confirmed_findings':[
            'Reward signs and terminal masking passed nine no-learning tests. All 800 game repetition counters agree with recorded endings.',
            'Both new best policies can prefer an immediate repetition draw over an available one-move checkmate. Pikafish independently confirmed no legal replies after the alternative mate moves.',
            'Only 0.386% and 0.482% of new-run training transitions have nonzero reward. Captures, progress and draws receive zero.',
            'The new runs started from scratch. Both selected checkpoints are episode 100: five visits per training start, with roughly 95% epsilon exploration at that point.',
            'New DQN last-minibatch loss increased sharply; new Double DQN did not show comparable loss growth.',
            'Repetition history is provided, but compressed by a pooled history network. Synthetic count-sensitivity probes show very small Q changes, not proof that the history encoder is mathematically broken.',
            'Both runs early-stopped before the 100000-step exploration decay finished.'
        ],
        'phase_0_no_training':[
            'Keep these diagnosis outputs and the no-learning reward tests.',
            'Add instrumentation in a future implementation: greedy score on held-out seeds from training starts, validation score, repetition rates, terminal-transition fraction, legal-Q range and immediate-win miss rate.',
            'Record a frozen untrained greedy baseline as well as the random baseline; current logs have only the latter.',
            'Treat the previously inspected ten boards as a regression bank. Reserve additional never-inspected starts for a later final test.'
        ],
        'future_ablations_not_started':[
            {'id':'signal','single_change':'Test a terminal-aware replay sampler with recorded sampling probabilities and importance weights; retain wins, losses and terminal draws rather than oversampling only wins. Keep the objective and algorithm unchanged.','reason':'Uniform batches rarely include nonzero reward; terminal draw examples are also needed to calibrate value zero.','success_checks':['Lower immediate-terminal Q error','Fewer missed one-move wins','Higher validation win rate without increased losses']},
            {'id':'history','single_change':'Test explicit per-action repetition information in the value representation, while retaining legal masks and correct history. Version the architecture and train from scratch with matched initialization/budgets.','reason':'The environment knows a move ends in a draw, but the learned value barely reacts to its repetition count.','success_checks':['Lower terminal-draw Q error','Reduced repetition on validation boards','No degradation in defensible draw positions']},
            {'id':'stability','single_change':'For DQN, test a lower learning rate in isolation and monitor legal Q against the sparse-return [-1,1] bound. Do not assume the same change is needed for Double DQN.','reason':'DQN logs show rising regression loss and some best-checkpoint draw predictions exceed 1.','success_checks':['Stable legal-Q and loss traces','Improved held-out-seed greedy performance','Validation improvement']},
            {'id':'reward_shaping','single_change':'If sparse credit remains a bottleneck, separately test bounded potential-based shaping: r_train = r_env + gamma*Phi(next) - Phi(current), with Phi=0 at true terminal states. Select Phi and its scale only on training/validation starts.','reason':'Give intermediate guidance without unconditional per-check, survival or reversible-move bonuses. The theorem does not guarantee neural training improvement.','success_checks':['No profitable reward loop','Original unshaped win rate improves','No loss of zero-sum environment rewards']}
        ],
        'experimental_controls':[
            'Each ablation changes one factor and compares DQN and Double DQN with matched settings, board order and evaluation seeds.',
            'Use more than one training seed before claiming an algorithm advantage.',
            'A separate warm-start study may initialize from each old best checkpoint. Label it as fine-tuning with a new optimizer/replay, not exact resume; full old resume state is not in best.pt.',
            'Coordinate the minimum learner-step budget, epsilon schedule and early-stop monitoring; do not blindly force more episodes when values already deteriorate.',
            'Do not add a blanket negative reward for every draw: drawing can be the correct defense and symmetric negative draw rewards would change the zero-sum payoff.',
            'Do not apply a one-move tactical rule only to one model and call it pure learned-policy improvement. If tested, label and compare it separately.',
            'An external time cutoff correctly bootstraps here. If the actual objective becomes winning within exactly 160 plies, explicitly create a finite-horizon variant, include remaining time in the observation, and use terminal targets consistently.'
        ],
        'uncertainty':'This audit establishes failure mechanisms and concrete decision errors. One seed per algorithm and only saved best weights cannot isolate every causal contribution or prove classic overfitting. Late checkpoints/replay tensors are not local; final replay reward counts were reconstructed from terminal-step CSV rows.',
        'sources':{
            'double_dqn':'https://arxiv.org/abs/1509.06461',
            'time_limits':'https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/',
            'potential_shaping':'https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf'
        }
    }
    write_json(OUT/'next-experiment-plan.json',proposal)
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Why the new models did not improve</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;color:#233b35;background:#f5f7f3}h1,h2{line-height:1.25}section{padding:22px;background:white;border:1px solid #d8e1d7;border-radius:12px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{padding:10px;border-bottom:1px solid #dbe2da;text-align:left}th{background:#edf2e9}.case{display:grid;grid-template-columns:minmax(230px,360px) 1fr;gap:24px;align-items:start}svg{width:100%;height:auto}code{overflow-wrap:anywhere}a{color:#17685c}.note{border-left:4px solid #b58134;padding:12px 16px;background:#fff8e9}@media(max-width:720px){.case{display:block}}</style>',
        '<h1>Why the new models did not improve</h1><p>25 October 2026. Diagnosis only: no training, no optimizer updates, and no changes to production rules or model weights.</p>',
        '<section><h2>Main finding</h2><p>The models misvalue concrete moves. Both new models have recorded positions where a one-move win was available, but they chose an immediate repetition draw. The draw rule and reward calculation worked as implemented. The learned action values did not reflect those outcomes.</p><p>Sparse feedback, weak use of repetition history, and limited per-start experience at the selected checkpoint are plausible contributors. DQN also shows loss growth. A single training seed does not prove the exact contribution of each factor.</p></section>',
        '<section><h2>Checks that passed</h2><ul><li>All 800 recorded games: position keys include side to move; repetition endings occur on the third appearance, with no earlier third appearance ignored.</li><li>Nine no-learning tests: wins/losses, Red- and Black-triggered repetition, no-capture draws, capture reward, illegal-action masking, Double DQN selection/evaluation, and terminal versus truncated targets.</li><li>Pikafish independently confirmed the two one-move wins shown below: zero legal replies after the alternative moves.</li><li>All four model files and the production rules/trainer/encoder are unchanged.</li></ul><p>The research rule is threefold-position repetition or 120 non-capture plies. It is not the full tournament perpetual-check/chase adjudicator. This simplification is shared by old and new models.</p></section>',
        '<section><h2>Training signal and experience</h2><div class="scroll"><table><tr><th>Model</th><th>Nonzero rewards / learner steps</th><th>Fraction</th><th>Best checkpoint</th><th>Visits per start at best</th></tr>']
    for name,d in models.items():
        visits=list(d['starts_seen_at_best'].values())
        parts.append(f'<tr><td>{name}</td><td>{d["nonzero_reward_transitions"]} / {d["total_steps"]}</td><td>{d["nonzero_reward_fraction"]:.3%}</td><td>Episode {d["best_episode"]}</td><td>{min(visits)} to {max(visits)}</td></tr>')
    parts += ['</table></div><p>Reward: win +1, loss -1, everything else 0, including capture and repetition draw. A zero draw reward is conventional, not inherently a bug. It provides little direct help in assigning credit to useful intermediate moves.</p><p>The new runs started from random weights, not from the old best models. At episode 100 each new training start had been visited five times; epsilon was about 0.95. At early stop, epsilon was still about 0.25 and the 100,000-step decay was unfinished. Early stopping did preserve the early best weights; the late, worse weights were not accidentally exported.</p><p>In the final replay window, reconstructed from CSV terminal step numbers, new DQN had 15 nonzero-reward transitions in 10,000; new Double DQN had 36. A uniform batch of 64 would contain only 0.096 and 0.230 such transitions on average. Zero-reward transitions can still propagate value, so these counts alone do not prove why learning failed.</p></section>',
        '<section><h2>Repetition in the 200-game tests</h2><div class="scroll"><table><tr><th>Model</th><th>Repetition draws</th><th>Red triggered final repeat</th><th>Black triggered final repeat</th></tr>']
    for name,d in models.items():
        s=d['test_repetition']
        parts.append(f'<tr><td>{name}</td><td>{s["games"]}/200 ({s["fraction"]:.1%})</td><td>{s["red_made_final_move"]}</td><td>{s["black_made_final_move"]}</td></tr>')
    parts += ['</table></div><p>New DQN did not have more repetition draws than old DQN overall (159 versus 161). New Double DQN did (153 versus 138). Therefore repetition is a common failure mode, not a complete explanation for every score difference.</p><p>All history counts are present in the observation. The network compresses the full history into a small pooled vector. In a synthetic probe that changed a relevant past occurrence count from two to one, chosen-action Q changed very little (median about 0.00032 for new DQN and 0.000058 for new Double DQN). This shows low sensitivity in the sampled states, not proof of a broken encoder.</p></section>']
    for case in cases:
        p,w=case['probe'],case['win']
        parts += [f'<section><h2>Concrete error: {case["name"]}</h2><div class="case">{case["svg"]}<div>',
            f'<p>Recorded decision at ply {p["red_ply"]}. The saved model chose the larger predicted Q value.</p>',
            '<table><tr><th>Move</th><th>Predicted Q</th><th>Actual immediate result</th></tr>',
            f'<tr><td>{p["move"]} (chosen)</td><td>{p["q"]:.4f}</td><td>Repetition draw: 0</td></tr>',
            f'<tr><td>{w["move"]} (available)</td><td>{w["q"]:.4f}</td><td>Checkmate: +1</td></tr></table>',
            '<p>The available win was independently checked with Pikafish. The winning branch is a counterfactual diagnostic, not a move made by the model.</p>',
            f'<p><a href="examples/{case["actual"]}" download>Actual recorded draw</a><br><a href="examples/{case["alternative"]}" download>Alternative winning branch</a></p><p>Both JSON files can be loaded in the website to inspect the moves. Do not use these test examples as training data.</p></div></div></section>']
    parts += ['<section><h2>Value stability</h2><p>Under the current reward scheme, true discounted returns are bounded by -1 and +1. New DQN assigned values above +1 to six of twelve sampled immediate-draw actions, although their exact return is zero. New Double DQN also assigned positive values to immediate draws, but none of its twelve sampled values exceeded +1.</p><p>For new DQN, the median logged last-minibatch loss rose from 0.0057 in episodes 1-100 to 2.306 in episodes 1001-1500. New Double DQN changed from 0.0055 to 0.0202. These are per-episode snapshots of the last batch, not full-loss averages. The DQN trace supports a value-stability concern; it does not prove the same instability in Double DQN.</p><p><a href="https://arxiv.org/abs/1509.06461">The Double DQN paper</a> addresses value overestimation. It does not guarantee better performance for every training run or fix sparse feedback by itself.</p></section>',
        '<section><h2>Proposed adjustment order — not started</h2><ol><li>Add diagnostics before spending another training budget: greedy training-start evaluation, validation, repetition, terminal-value error, Q range and immediate-win misses. Keep random and untrained-greedy baselines separate.</li><li>Test replay sampling that retains informative terminal events, including draws, with sampling probabilities and importance weights. Compare it against the current sampler while leaving other settings fixed.</li><li>Separately test explicit action-level repetition information. Preserve legal masks and the full rule state; version any new network architecture.</li><li>For DQN, separately test a smaller learning rate and monitor Q stability. Do not change both algorithms blindly based on DQN loss alone.</li><li>If needed, test potential-based intermediate rewards in a separate experiment. Keep the original evaluation payoff and terminal semantics unchanged.</li></ol><p>Coordinate early stopping with the exploration schedule and use matched learner-step budgets. Warm-starting old policies is a separate fine-tuning experiment, not exact resume. Do not change all settings at once.</p><p>Potential shaping uses <code>r_train = r_env + gamma * Phi(next) - Phi(current)</code>, with zero terminal potential. It avoids arbitrary repeatable move bonuses under its assumptions; it does not guarantee better neural training. See <a href="https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf">Ng, Harada and Russell</a>.</p><p>Do not blindly penalize all draws or reward every check/capture. A draw can be the correct defense; symmetric negative draw payoffs alter the zero-sum objective. A one-move rule override, if tested, must be labeled as a separate hybrid policy.</p></section>',
        '<section><h2>Limits and next evaluation</h2><p>No classic overfitting diagnosis is claimed: greedy performance on training starts was not logged, and the late full checkpoints are not local. The finite sample and one training seed cannot isolate all causes. The old models also show some of these weaknesses.</p><p>The current 160-ply cap is an external cutoff and correctly bootstraps. A task explicitly defined as winning within 160 plies would need remaining-time observations and consistent finite-horizon terminal targets. See <a href="https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/">Gymnasium time-limit guidance</a>.</p><p>The old ten-board test bank has now been inspected repeatedly. Keep it for regression checks; reserve never-inspected boards for a later final evaluation. No model or reward changes should be tuned directly on its recorded positions.</p><p><a href="diagnosis.json">Full measurements</a> · <a href="next-experiment-plan.json">Disabled experiment plan</a> · <a href="pikafish-verification.json">Engine verification</a> · <a href="reward-tests.log">No-learning tests</a></p></section></html>']
    (OUT/'report.html').write_text('\n'.join(parts),encoding='utf-8')
    shutil.copy2(ROOT/'.setup/reward-diagnostics-tests.log',OUT/'reward-tests.log')
    repro=OUT/'reproduce'
    repro.mkdir(exist_ok=True)
    for path in [ROOT/'tools/diagnose_rook_models.py',Path(__file__),ROOT/'xiangqi_lab/tests/test_reward_diagnostics.py']:
        shutil.copy2(path,repro/path.name)
    checks=dict(data['checks'],reward_tests_passed=9,engine_verified_mate_examples=2,
        production_sources_unchanged=all(sha(ROOT/'xiangqi_lab'/n)==h for n,h in data['source_hashes'].items()),
        no_training=True,no_model_changes=True)
    write_json(OUT/'verification.json',checks)
    bundle=OUT.parent/'Xiangqi_Training_Diagnosis_20261025.zip'
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file(): z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with zipfile.ZipFile(bundle) as z: assert z.testzip() is None
    print(json.dumps(checks,indent=2))
    print(bundle)


if __name__=='__main__':
    main()
