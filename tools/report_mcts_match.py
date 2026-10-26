"""Build a standalone results page with charts and a saved-game viewer."""
from pathlib import Path
from html import escape
import argparse
import csv
import json

ROOT=Path(__file__).resolve().parents[1]
LABELS={'double_dqn_win':'Double DQN win','mcts_v4_win':'MCTS win','draw':'Draw','ply_limit':'Move limit'}
COLORS={'double_dqn_win':'#36739b','mcts_v4_win':'#bd7130','draw':'#b4c1ca','ply_limit':'#8075ad'}
SUITES={'existing10':'10 existing test positions','familiar3':'3 familiar presets'}
GROUPS={'rook':'Rook and pawn','cannon':'Cannon and pawn','horse':'Horse and pawn'}
NAMES={'double_dqn':'Double DQN','mcts_v4':'MCTS v4'}
def chart(counts):
    total=counts['games'];left=20
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1020 125" role="img" aria-label="Outcome counts">']
    for key in COLORS:
        n=counts[key];width=980*n/total
        if n:
            parts.append(f'<rect x="{left}" y="10" width="{width}" height="55" fill="{COLORS[key]}"/>')
            parts.append(f'<text x="{left+width/2}" y="45" text-anchor="middle" fill="#152c38" font-family="Arial" font-size="25" font-weight="bold">{n}</text>')
        left+=width
    for i,key in enumerate(COLORS):
        x=20+i*250
        parts.append(f'<rect x="{x}" y="91" width="17" height="17" fill="{COLORS[key]}"/><text x="{x+25}" y="106" font-family="Arial" font-size="17">{LABELS[key]}: {counts[key]}</text>')
    parts.append('</svg>');return ''.join(parts)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    out=args.directory.resolve();summary=json.loads((out/'summary.json').read_text('utf-8'))
    assert summary['status']=='complete'
    rows=list(csv.DictReader((out/'games.csv').open(encoding='utf-8-sig',newline='')))
    episodes=[json.loads((out/r['replay']).read_text('utf-8')) for r in rows]
    panels=[]
    for suite,label in SUITES.items():
        counts=summary['suites'][suite]
        tablerows=''.join('<tr>'+''.join(f'<td>{v}</td>' for v in (GROUPS[g],stats['games'],stats['double_dqn_win'],stats['mcts_v4_win'],stats['draw'],stats['ply_limit']))+'</tr>' for g,stats in summary['by_group'][suite].items())
        note=('Each of the ten earlier test starts was played twice, with the model assignments swapped. These are reused regression positions, not a new blind test. MCTS used search without stored experience on these custom starts.' if suite=='existing10' else 'Each preset was played twice, with the model assignments swapped. MCTS received the matching isolated experience section. These familiar positions are not evidence of unseen-position generalisation.')
        panels.append(f'<section><h2>{label}</h2><p>{note}</p><div class="score"><strong>{counts["double_dqn_win"]}</strong> Double DQN wins &nbsp; / &nbsp; <strong>{counts["mcts_v4_win"]}</strong> MCTS wins &nbsp; / &nbsp; {counts["draw"]} draws &nbsp; / &nbsp; {counts["ply_limit"]} move limits</div>{chart(counts)}<table><thead><tr><th>Endgame</th><th>Games</th><th>Double DQN wins</th><th>MCTS wins</th><th>Draws</th><th>Limits</th></tr></thead><tbody>{tablerows}</tbody></table></section>')
        (out/f'{suite}_results.svg').write_text(chart(counts),encoding='utf-8')
    gamerows=[]
    for i,r in enumerate(rows):
        cells=[SUITES[r['suite']],r['board'],NAMES[r['red']],LABELS[r['result']],r['reason'],r['plies'],r['mcts_prior_hits']]
        gamerows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in cells)+f'<td><button class="view-game" data-index="{i}">View</button> <a href="{escape(r["replay"])}">JSON</a></td></tr>')
    plan=json.loads((out/'plan.json').read_text('utf-8'))
    reasons=', '.join(f'{k}: {v}' for k,v in summary['reasons'].items())
    inf=summary['inference'];sim=summary['mcts_simulations']
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Double DQN vs MCTS v4</title><style>
*{box-sizing:border-box}body{font:16px/1.6 system-ui,Arial,sans-serif;margin:0;background:#f3f5f4;color:#23383e}main{max-width:1140px;margin:35px auto;padding:0 22px 50px}h1{font-size:34px;margin-bottom:8px}h2{font-size:23px}h3{font-size:18px}section{background:white;border:1px solid #d9e1df;border-radius:12px;padding:24px;margin:24px 0}.lead{max-width:900px;color:#4b5b60}.score{background:#edf2f3;border-radius:8px;padding:14px;font-size:19px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px 8px;border-bottom:1px solid #dce2e2}th{color:#354f58}svg{width:100%;height:auto}.scroll{overflow:auto}button,select,input{font:inherit}button,select{padding:8px 12px;border:1px solid #b6c4c7;border-radius:7px;background:#f8faf9;color:#263f46}button{cursor:pointer}a{color:#245c87}.badge{display:inline-block;background:#e2eee7;color:#2b5940;padding:4px 10px;border-radius:6px;margin-right:6px}.viewer{display:grid;grid-template-columns:minmax(260px,520px) 1fr;gap:26px}.controls{display:flex;gap:8px;margin:10px 0}.viewer select{width:100%}#seek{width:100%}code{font-size:13px;overflow-wrap:anywhere}.issue{border-left:4px solid #bd7130;padding-left:16px}.small{font-size:14px;color:#53646a}#fen{word-break:break-all}.legend{color:#53646a}@media(max-width:750px){.viewer{grid-template-columns:1fr}section{padding:16px}h1{font-size:28px}}</style></head><body><main>
<h1>Old Double DQN vs received MCTS v4</h1><p class="lead">Frozen-model match on 26 games. The ten-position regression test and the three familiar presets are reported separately. This is a small comparison of the delivered agents, not a ranking of all Double DQN and MCTS algorithms.</p>
<p><span class="badge">No training</span><span class="badge">Files unchanged</span><span class="badge">All replays checked</span></p>
__PANELS__
<section><h2>How to read these results</h2>
<p class="issue"><strong>Known issue in the received MCTS code.</strong> A forced one-ply checkmate rollout returns −1 from the starting player's perspective instead of +1. The delivered search code was left unchanged for this match. This can affect move scoring; its effect on the final win count has not been measured. A repaired version needs a separate comparison.</p>
<p>All three supplied folders contain scene-based experience. The user selected the latest v4 before the match. Its code is the same as v2 and v3; its experience tables differ. This is MCTS with stored state-action values, not a neural-network checkpoint.</p>
<p>Both agents' stored learning experience is from Red play. Double DQN uses its existing rotated, colour-swapped view as Black. MCTS searches natively for either side; its stored entries contain Red actions only. Loading a scene for Black does not add defensive training.</p>
<p>Each game starts a fresh MCTS instance. Familiar presets receive their scene ID explicitly even when MCTS first acts at ply 1. Custom positions keep the received search-only behaviour. Reading stored values is not learning; the experience file and loaded table stay unchanged.</p>
<details><summary>Search budget, seeds and integrity checks</summary><p>Move cap: 160 plies. Repetition and 120 non-capture plies are draws; reaching the move cap is a separate truncation. Stalemate is a loss under Xiangqi rules.</p><p>MCTS uses the delivered inference defaults: at most 96 simulations, rollout depth 10, and an 800 ms requested move budget. The source reserves 20% of this budget and requires at least four simulations. Its training configuration mentions 500 iterations; those training-time settings were not substituted into the delivered inference policy.</p><p>One fixed search seed is assigned per board and reused for the two colour assignments. Time-based stopping can vary with computer load, so fixed seeds alone do not guarantee identical reruns. Actual simulation counts and move times are saved in every replay. A search budget is not equal compute to a neural forward pass.</p><p>__PERFORMANCE__</p><p>Source code, experience JSON and Double DQN checkpoint hashes were checked before and after. Double DQN tensors and loaded MCTS tables were unchanged. All games were replayed with the shared rules. No model or hyperparameter was selected using these results.</p></details>
<p><a href="summary.json">Summary and checks</a> · <a href="games.csv">CSV results</a> · <a href="plan.json">Saved test plan</a> · <a href="inspection.json">Received package inspection</a></p>
<details><summary>Run audit</summary><p>The first 20 games were preserved after the checker mistook MCTS's first memory load for a change to that memory. Only the checker was corrected. The unfinished first preset game was discarded and restarted with the same seed and settings. All 26 final games passed replay and integrity checks. See the <a href="resume_audit.json">resume record</a> and <a href="aborted_partial_audit.json">discarded partial run</a>.</p></details>
</section>
<section id="replay"><h2>Watch a saved game</h2><div class="viewer"><div><svg id="board" viewBox="0 0 500 560" role="img" aria-label="Saved Xiangqi position"></svg></div><div><label for="game">Game</label><select id="game"></select><p id="game-info"></p><div class="controls"><button id="first">Start</button><button id="prev">Back</button><button id="next">Next</button><button id="last">End</button></div><input id="seek" type="range" min="0" value="0"><p id="step-info"></p><p id="move-info" class="small"></p><p id="fen" class="small"></p><p><a id="download" href="#">Open full replay JSON</a></p><p class="small">This viewer replays saved moves only. It does not run a model or train anything. The same JSON can be opened with Load game in Xiangqi Lab.</p></div></div></section>
<section><h2>Every game</h2><p class="small">Red column identifies the Red player; the other agent plays Black. Memory hits count stored values consulted during search, including repeated lookups, not unique positions.</p><div class="scroll"><table><thead><tr><th>Set</th><th>Board</th><th>Red</th><th>Result</th><th>Ending</th><th>Plies</th><th>Memory hits</th><th>Replay</th></tr></thead><tbody>__ROWS__</tbody></table></div><p>Endings: __REASONS__.</p></section>
<section><h2>Earlier PPO match</h2><p>On the same ten-position set, the earlier frozen Double DQN versus PPO seed 42 match finished with Double DQN 0 wins, PPO 2 wins and 18 draws. That was a different opponent and a deterministic match. Do not combine those games with this MCTS test or infer a direct PPO versus MCTS ranking.</p></section>
<p class="small">Local evaluation on 26 October 2026. Original Double DQN checkpoint: best episode 200. Received MCTS: v4, original policy and experience preserved.</p>
</main><script>
const rows=__JSON_ROWS__,episodes=__JSON_EPISODES__;
const gameSelect=document.getElementById('game'),seek=document.getElementById('seek');
const labels={K:'帥',A:'仕',E:'相',H:'馬',R:'車',C:'炮',P:'兵',k:'將',a:'士',e:'象',h:'馬',r:'車',c:'砲',p:'卒'};
let active=0,ply=0;
rows.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${r.suite} · ${r.board} · Red: ${r.red}`;gameSelect.appendChild(o)});
function svgBoard(fen,last){let bits=['<rect x="10" y="10" width="480" height="535" rx="12" fill="#f4ead6"/>'];const line=(x,y,x2,y2)=>bits.push(`<line x1="${x}" y1="${y}" x2="${x2}" y2="${y2}" stroke="#b5a17d"/>`);for(let r=0;r<10;r++)line(50,40+r*50,450,40+r*50);for(let c=0;c<9;c++){let x=50+c*50;if(c===0||c===8)line(x,40,x,490);else{line(x,40,x,240);line(x,290,x,490)}}for(let y of [40,390]){line(200,y,300,y+100);line(300,y,200,y+100)}bits.push('<text x="250" y="273" font-size="17" text-anchor="middle" fill="#8c7754">楚 河　漢 界</text>');if(last){for(let s of [last.slice(0,2),last.slice(2)]){let x=50+(s.charCodeAt(0)-97)*50,y=40+(9-Number(s[1]))*50;bits.push(`<rect x="${x-23}" y="${y-23}" width="46" height="46" rx="6" fill="#86a69b" opacity=".28"/>`)}}fen.split(' ')[0].split('/').forEach((rank,r)=>{let c=0;for(let p of rank){if(/[1-9]/.test(p)){c+=Number(p);continue}let x=50+c*50,y=40+r*50,color=p===p.toUpperCase()?'#b3473b':'#334c53';bits.push(`<circle cx="${x}" cy="${y}" r="21" fill="#fff9ea" stroke="${color}"/><text x="${x}" y="${y+9}" font-size="28" font-family="serif" text-anchor="middle" fill="${color}">${labels[p]}</text>`);c++}});for(let c=0;c<9;c++)bits.push(`<text x="${50+c*50}" y="526" font-size="12" text-anchor="middle" fill="#7b6b52">${String.fromCharCode(97+c)}</text>`);for(let r=0;r<10;r++)bits.push(`<text x="25" y="${44+r*50}" font-size="12" fill="#7b6b52">${9-r}</text>`);return bits.join('')}
function draw(){let ep=episodes[active],r=rows[active],m=ply?ep.moves[ply-1]:null,fen=m?m.fen_after:ep.initial_fen;document.getElementById('board').innerHTML=svgBoard(fen,m?.move);seek.max=ep.moves.length;seek.value=ply;document.getElementById('step-info').textContent=`Ply ${ply} / ${ep.moves.length} · ${fen.split(' ')[1]==='b'?'Black':'Red'} to move`;document.getElementById('game-info').textContent=`${r.result.replaceAll('_',' ')} · ${r.reason} · seed ${r.seed}`;document.getElementById('move-info').textContent=m?`${m.actor}: ${m.move} · ${m.analysis.elapsed_ms} ms`:'Initial position';document.getElementById('fen').textContent=fen;document.getElementById('download').href=r.replay;document.getElementById('prev').disabled=ply===0;document.getElementById('next').disabled=ply===ep.moves.length}
function choose(i){active=Number(i);ply=0;gameSelect.value=active;draw()}
gameSelect.addEventListener('change',()=>choose(gameSelect.value));seek.addEventListener('input',()=>{ply=Number(seek.value);draw()});document.getElementById('first').onclick=()=>{ply=0;draw()};document.getElementById('last').onclick=()=>{ply=episodes[active].moves.length;draw()};document.getElementById('prev').onclick=()=>{ply=Math.max(0,ply-1);draw()};document.getElementById('next').onclick=()=>{ply=Math.min(episodes[active].moves.length,ply+1);draw()};document.querySelectorAll('.view-game').forEach(b=>b.onclick=()=>{choose(b.dataset.index);document.getElementById('replay').scrollIntoView({behavior:'smooth'})});choose(0);
</script></body></html>'''
    performance=f"Completed MCTS simulations: {sim['total']:,}; mean per MCTS move: {sim['mean_per_move']:.1f}. Mean move time: MCTS {inf['mcts_v4']['mean_ms']:.1f} ms, Double DQN {inf['double_dqn']['mean_ms']:.1f} ms. These timings describe this local run only."
    replacements={'__PANELS__':''.join(panels),'__ROWS__':''.join(gamerows),'__REASONS__':escape(reasons),'__PERFORMANCE__':performance,
                  '__JSON_ROWS__':json.dumps(rows,ensure_ascii=False).replace('<','\\u003c'),
                  '__JSON_EPISODES__':json.dumps(episodes,ensure_ascii=False).replace('<','\\u003c')}
    for a,b in replacements.items():html=html.replace(a,b)
    (out/'report.html').write_text(html,encoding='utf-8')
    print(out/'report.html')

if __name__=='__main__':main()
