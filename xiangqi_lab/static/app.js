'use strict';
// Keep developer examples available to the API and CLI, but out of player menus.
const HIDDEN_MODEL_IDS=new Set(['custom-example','custom-team-template']);
// Display names are independent of model IDs and saved game data.
const EN={
  'Starting position':'Full board',
  'Cannon coordination':'Cannon & pawn',
  'Starter · 6 pieces':'6 pieces',
  'Explore · 8 pieces':'8 pieces',
  'Reference · 32 pieces':'32 pieces',
  'Human':'Player',
  'Random legal policy':'Random',
  'JSON example (untrained)':'Example (not trained)',
  'Team template (untrained)':'Team demo (not trained)',
  'Bundled NNUE reference engine.':'Built-in Xiangqi AI.',
  'Untrained baseline.':'Picks a random valid move.',
  'JSON adapter example with random legal moves.':'Example code that picks a random move.',
  'Drop-in Python policy; replace policy.py with your inference code.':'A working example for your team to build on.'
};
for(const p of 'KAEHRCPkaehrcp'){
  EN['symbol.'+p]=p.toUpperCase();
  EN['piece.'+p]={K:'King',A:'Advisor',E:'Elephant',H:'Horse',R:'Rook',C:'Cannon',P:'Pawn'}[p.toUpperCase()];
}
let lang='en',dictionary={},modelCheck=null,boardSymbols={};
const localeCache={};
async function loadLocale(language){
  if(!localeCache[language]){
    const response=await fetch('/locales/'+language+'.json');
    if(!response.ok)throw Error('Locale unavailable');
    localeCache[language]=await response.json();
  }
  return localeCache[language];
}
function t(key,values={}){return String((lang==='en'?null:dictionary[key])??EN[key]??key).replace(/\{(\w+)\}/g,(_,name)=>values[name]??'{'+name+'}');}
function pieceSymbol(p){return boardSymbols['symbol.'+p]??EN['symbol.'+p];}
function pieceName(p){return t('piece.'+p);}
function localizeError(message){
  if(!message)return '';
  if(dictionary[message])return t(message);
  for(const prefix of ['Model returned an illegal move: ','Model inference failed: ','Model process failed: ','Model exited unexpectedly: ']){
    if(message.startsWith(prefix))return t(prefix)+message.slice(prefix.length);
  }
  return message;
}
async function loadLanguage(value){
  $('notice').hidden=true;
  lang=(meta.languages||['en']).includes(value)?value:'en';
  dictionary={};
  if(lang!=='en'){
    try{dictionary=await loadLocale(lang);}
    catch{lang='en';}
  }
  try{localStorage.setItem('xiangqi-language',lang);}catch{}
  document.documentElement.lang=lang;
  document.title=t('Xiangqi Lab');
  document.querySelectorAll('[data-i18n]').forEach(el=>{const key=el.dataset.i18n||el.textContent;el.dataset.i18n=key;el.textContent=t(key);});
  document.querySelectorAll('[data-piece]').forEach(el=>{el.textContent=pieceSymbol(el.dataset.piece);});
  for(const [data,attr]of [['i18nTitle','title'],['i18nAria','aria-label']])document.querySelectorAll('[data-'+data.replace('i18n','i18n-').toLowerCase()+']').forEach(el=>el.setAttribute(attr,t(el.dataset[data])));
  $('language').value=lang;
  $('connection').textContent=t('Local');
  populateModels();populatePresets();renderModelList();
  if(state){const fen=$('fen-input').value;render();$('fen-input').value=fen;if(editing)renderPalette();}
}
function visibleModels(){return meta.models.filter(m=>!HIDDEN_MODEL_IDS.has(m.id));}
function populateModels(){
  const choices=visibleModels();
  for(const side of ['red','black']){const select=$(side+'-model'),previous=select.value;select.replaceChildren();for(const model of choices){const o=document.createElement('option');o.value=model.id;o.textContent=modelName(model.id);select.appendChild(o);}select.value=choices.some(m=>m.id===previous)?previous:(side==='red'?'human':meta.engine_available?'pikafish':'random');}
}
function populatePresets(){
  $('presets').replaceChildren();
  for(const p of meta.presets){const b=document.createElement('button');b.className='preset';b.dataset.id=p.id;for(const [tag,text]of [['strong',p.name],['span',p.tag]]){const e=document.createElement(tag);e.textContent=t(text);b.appendChild(e);}b.onclick=async()=>{if(editing)endEditor();await resetTo(p.fen,p.name,p.id);};$('presets').appendChild(b);}
}
function renderModelList(){
  $('model-list').replaceChildren();
  for(const model of visibleModels().filter(m=>m.id!=='human')){
    const row=document.createElement('div');row.className='model-row';
    const text=document.createElement('div'),name=document.createElement('strong'),description=document.createElement('small');name.textContent=modelName(model.id);description.textContent=t(model.description);text.append(name,description);
    const button=document.createElement('button');button.className='button secondary check-model';button.textContent=t('Test');button.setAttribute('aria-label',t('Test {model}',{model:modelName(model.id)}));button.disabled=busy;button.onclick=()=>validateModel(model.id);row.append(text,button);$('model-list').appendChild(row);
  }
  $('model-check-status').textContent=!modelCheck?'':modelCheck.message?t(modelCheck.message):modelCheck.error?localizeError(modelCheck.error):t(modelCheck.passed?'Passed: {model}, 3 test positions. Your game is unchanged.':'Testing {model}…',{model:modelName(modelCheck.model)});
}
async function validateModel(model){
  if(busy)return;stop();setBusy(true);modelCheck={model};renderModelList();
  try{const result=await api('models/check',{model});modelCheck={model,passed:result.passed};}
  catch(e){modelCheck={model,error:e.message};}
  finally{setBusy(false);renderModelList();}
}
const $=id=>document.getElementById(id);
const PIECES='KAEHRCPkaehrcp';
const NS='http://www.w3.org/2000/svg';
let state, meta, session, selected=null, flipped=false, running=false, busy=false, editing=false, draft=null, palette='R', title='Rook & pawn', presetId='rook', timer=null;
let noticeTimer=null;
function notify(message,error=false){ clearTimeout(noticeTimer); $('notice').textContent=localizeError(message); $('notice').hidden=false; $('notice').className='notice'+(error?'':' success'); if(!error)noticeTimer=setTimeout(()=>$('notice').hidden=true,4500); }
async function api(path,data={}){
  const response=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session,revision:state?.revision,...data})});
  const result=await response.json();
  if(!response.ok){if(result.state){state=result.state;render();}throw Error(result.error||'The local service is unavailable.');}
  return result;
}
function control(side){return $(side+'-model').value;}
function modelName(id){return t(meta.models.find(m=>m.id===id)?.name||id);}
function setBusy(value){busy=value;document.body.classList.toggle('busy',value);renderControls();}
function stop(){running=false;clearTimeout(timer);timer=null;renderControls();}
function renderControls(){
  if(!state)return;
  const done=!!state.outcome;
  ['reload-models','undo','reset','load-fen','edit-toggle','import','export','red-model','black-model','budget','ply-limit','edit-apply','edit-cancel','clear-board'].forEach(id=>$(id).disabled=busy);
  document.querySelectorAll('.preset,.check-model').forEach(e=>e.disabled=busy);
  $('undo').disabled=busy||editing||!state.history.length;
  $('reset').disabled=busy||editing;
  $('load-fen').disabled=busy||editing;
  $('import').disabled=busy||editing;
  $('export').disabled=busy||editing;
  $('step').disabled=busy||editing||done||control(state.side)==='human';
  $('play').disabled=editing||done||(busy&&!running);
  $('play').textContent=running?t("Pause"):busy?t("Please wait…"):t("Play");
  $('play').style.background=running?'#8b6a48':'';
  $('edit-toggle').textContent=editing?t("Editing board"):t("Edit board");
  $('edit-toggle').disabled=busy||editing;
  $('red-label').textContent=modelName(control('red'));
  $('black-label').textContent=modelName(control('black'));
  for(const side of ['red','black'])$(side+'-state').textContent=done?t('Game over'):state.side===side?(busy?t('Thinking…'):control(side)==='human'?t('Your turn'):t('AI turn')):t('Waiting');
  if(editing)$('run-status').textContent=t("Click Use this board when ready.");
  else if(done)$('run-status').textContent=outcomeText();
  else if(busy)$('run-status').textContent=running?t("Thinking… Pause stops after this move."):t("Finishing this move…");
  else if(running)$('run-status').textContent=control(state.side)==='human'?t("Your turn. Click a piece."):t("AI is playing.");
  else $('run-status').textContent=t('Red: {red} · Black: {black}',{red:modelName(control('red')),black:modelName(control('black'))});
}
function outcomeText(){
  if(!state.outcome)return '';
  const reasons={checkmate:t("Checkmate"),stalemate:t("Stalemate"),repetition:t("Same position 3 times"),no_capture:t("120 moves without a capture"),ply_limit:t("Move limit reached")};
  const o=state.outcome;
  return (o.truncated?t("Game stopped"):o.winner?(o.winner==='red'?t("Red wins"):t("Black wins")):t("Draw"))+' · '+reasons[o.reason];
}
function svg(tag,attrs={},parent=$('board')){const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.appendChild(e);return e;}
function pos(r,c){return flipped?[46+(8-c)*56,48+(9-r)*56]:[46+c*56,48+r*56];}
function sq(r,c){return 'abcdefghi'[c]+(9-r);}
function rc(s){return [9-Number(s[1]),'abcdefghi'.indexOf(s[0])];}
function drawBoard(){
  const board=$('board');board.replaceChildren();
  const defs=svg('defs');const grad=svg('radialGradient',{id:'piece-bg',cx:'35%',cy:'25%',r:'80%'},defs);
  svg('stop',{offset:'0%', 'stop-color':'#fff9e9'},grad);svg('stop',{offset:'100%','stop-color':'#ecdbb9'},grad);
  const shadow=svg('filter',{id:'piece-shadow',x:'-30%',y:'-30%',width:'160%',height:'170%'},defs);svg('feDropShadow',{dx:'0',dy:'2',stdDeviation:'1.3','flood-color':'#765a32','flood-opacity':'.2'},shadow);
  for(let r=0;r<10;r++){svg('line',{x1:46,y1:48+r*56,x2:494,y2:48+r*56,class:'board-line'});const label=svg('text',{x:21,y:52+r*56,fill:'#b4a183','font-size':11,'text-anchor':'middle'});label.textContent=flipped?r:9-r;}
  for(let c=0;c<9;c++){
    const x=46+c*56;
    svg('line',{x1:x,y1:48,x2:x,y2:272,class:'board-line'});svg('line',{x1:x,y1:328,x2:x,y2:552,class:'board-line'});
    if(c===0||c===8)svg('line',{x1:x,y1:272,x2:x,y2:328,class:'board-line'});
    const label=svg('text',{x,y:581,fill:'#b4a183','font-size':11,'text-anchor':'middle'});label.textContent='abcdefghi'[flipped?8-c:c];
  }
  svg('rect',{x:42,y:44,width:456,height:512,fill:'none',stroke:'#b29a72','stroke-width':1.5});
  for(const r of [0,7]){svg('path',{d:`M 214 ${48+r*56} L 326 ${48+(r+2)*56} M 326 ${48+r*56} L 214 ${48+(r+2)*56}`,class:'board-line'});}
  for(const [r,c]of [[2,1],[2,7],[7,1],[7,7],...[3,6].flatMap(r=>[0,2,4,6,8].map(c=>[r,c]))]){
    const x=46+c*56,y=48+r*56;
    for(const dx of [-1,1])for(const dy of [-1,1])if(c+dx>=0&&c+dx<=8)svg('path',{d:`M ${x+dx*4} ${y+dy*11} L ${x+dx*4} ${y+dy*4} L ${x+dx*11} ${y+dy*4}`,class:'board-line'});
  }
  const river=svg('text',{x:270,y:308,fill:'#ae9672','text-anchor':'middle','font-size':22,'font-family':'KaiTi, STKaiti, serif','letter-spacing':boardSymbols['R I V E R']?9:4});river.textContent=boardSymbols['R I V E R']??'R I V E R';
  if(!editing&&state.history.length){const last=state.history.at(-1).move;for(const s of [last.slice(0,2),last.slice(2)]){const [x,y]=pos(...rc(s));svg('rect',{x:x-25,y:y-25,width:50,height:50,rx:8,fill:'#729274',opacity:.18});}}
  const targets=new Set(!editing&&selected?state.legal_moves.filter(m=>m.startsWith(selected)).map(m=>m.slice(2)):[]);
  const displayed=editing?draft:state.board;
  for(let r=0;r<10;r++)for(let c=0;c<9;c++){
    const name=sq(r,c),p=displayed[r][c],[x,y]=pos(r,c),isRed=p&&p===p.toUpperCase();
    const group=svg('g',{class:'piece-g',role:'button',tabindex:0,'aria-label':`${name} ${p?t(isRed?'Red':'Black')+' '+pieceName(p):t('Empty square')}`,'data-square':name});
    svg('rect',{x:x-27,y:y-27,width:54,height:54,fill:'transparent'},group);
    if(p){
      svg('circle',{cx:x,cy:y,r:24,fill:'url(#piece-bg)',stroke:'#cdb78d','stroke-width':1.2,filter:'url(#piece-shadow)',class:'piece-face'},group);
      svg('circle',{cx:x,cy:y,r:20.5,fill:'none',stroke:isRed?'#b96550':'#697158','stroke-width':.8,opacity:.75,class:'piece-ring'},group);
      const text=svg('text',{x,y:y+10,'text-anchor':'middle',fill:isRed?'#b14f38':'#46513e',class:'piece-text','font-size':pieceSymbol(p)===p.toUpperCase()?25:29},group);text.textContent=pieceSymbol(p);
    }
    if(selected===name&&!editing)svg('circle',{cx:x,cy:y,r:26,fill:'none',stroke:'#648463','stroke-width':3},group);
    if(targets.has(name))svg('circle',{cx:x,cy:y,r:p?25:6,fill:p?'none':'#618866',stroke:p?'#618866':'none','stroke-width':3,opacity:.8},group);
    group.addEventListener('click',()=>onSquare(r,c));group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();onSquare(r,c);}});
  }
}
function render(){
  if(!state)return;
  $('fen-input').value=state.fen;$('position-title').textContent=editing?t("Your board"):t(title);
  $('turn-badge').textContent=editing?t("Editing"):state.outcome?outcomeText():(state.side==='red'?t("Red to move"):t("Black to move"))+(state.in_check?t(" · In check"):'');
  $('turn-badge').className='turn-badge'+(state.side==='black'?' black-turn':'');
  $('piece-count').textContent=state.piece_count;
  $('ply-count').replaceChildren(document.createTextNode(state.history.length));const total=document.createElement('span');total.textContent=' / '+state.max_plies;$('ply-count').appendChild(total);
  $('legal-count').textContent=state.outcome?'0':state.legal_moves.length;
  const latest=state.history.at(-1)?.analysis||{};
  $('thinking-time').textContent=latest.elapsed_ms!==undefined?(latest.elapsed_ms/1000).toFixed(2)+'s':'—';
  const evaluation=state.history.slice().reverse().find(m=>m.analysis?.depth)?.analysis;
  $('evaluation').textContent=evaluation?t('Depth {depth} · {nodes} nodes',{depth:evaluation.depth,nodes:Number(evaluation.nodes||0).toLocaleString()}):t("Details appear after an AI move.");
  $('record-count').textContent=state.history.length;
  if(state.history.length){
    $('moves').replaceChildren();
    const rows=[];
    for(let i=0;i<state.history.length;i++){
      const m=state.history[i];
      if(!rows.length||m.side==='red')rows.push({red:null,black:null});
      rows.at(-1)[m.side]={...m,index:i};
    }
    rows.forEach((row,i)=>{const div=document.createElement('div');div.className='move-row';const n=document.createElement('span');n.textContent=String(i+1).padStart(2,'0');div.appendChild(n);for(const side of ['red','black']){const e=document.createElement('span'),m=row[side];e.className='move-cell '+(side==='red'?'red-move ':'')+(m?.index===state.history.length-1?'latest':'');e.textContent=m?m.move:'·';if(m)e.title=`${pieceName(m.piece)} ${m.move} · ${modelName(m.actor)}`;div.appendChild(e);}$('moves').appendChild(div);});
    $('moves').scrollTop=$('moves').scrollHeight;
  }else {const empty=document.createElement('div');empty.className='empty-log';const text=document.createElement('strong');text.textContent=t('No moves yet');empty.appendChild(text);$('moves').replaceChildren(empty);}
  $('board-hint').textContent=editing?t("Pick a piece, then click the board. Use × to remove a piece."):state.outcome?outcomeText():selected?t('{square} selected. Click a marked spot.',{square:selected}):control(state.side)==='human'?t("Click a piece to see where it can move."):t("This is the AI's turn. Click Play or AI move.");
  if(state.outcome)running=false;
  document.querySelectorAll('.preset').forEach(el=>el.classList.toggle('selected',el.dataset.id===presetId));
  renderControls();drawBoard();
}
async function action(path,data={}){
  if(busy)return false;
  setBusy(true);
  try{const result=await api(path,data);if(result.state)state=result.state;selected=null;$('notice').hidden=true;render();return true;}
  catch(e){stop();notify(e.message,true);return false;}
  finally{setBusy(false);}
}
async function onSquare(r,c){
  if(busy)return;
  if(editing){draft[r][c]=palette;drawBoard();return;}
  if(state.outcome)return;
  if(control(state.side)!=='human'){notify(t("This is the AI's turn. Click Play or AI move."));return;}
  const s=sq(r,c),piece=state.board[r][c];
  if(selected&&state.legal_moves.includes(selected+s)){
    if(await action('move',{move:selected+s}))schedule();return;
  }
  if(piece&&((piece===piece.toUpperCase())===(state.side==='red')))selected=selected===s?null:s;
  else selected=null;
  render();
}
function schedule(){
  clearTimeout(timer);
  if(running&&!busy&&!editing&&!state.outcome&&control(state.side)!=='human')timer=setTimeout(engineStep,450);
  renderControls();
}
async function engineStep(){
  if(busy||editing||state.outcome||control(state.side)==='human')return;
  const ok=await action('engine',{model:control(state.side),milliseconds:Number($('budget').value)});
  if(ok)schedule();
}
async function resetTo(fen,newTitle,newPreset){
  stop();
  const ok=await action('reset',{fen,max_plies:Number($('ply-limit').value)});
  if(ok){title=newTitle;presetId=newPreset;render();}
  return ok;
}
function startEditor(){
  if(busy)return;stop();editing=true;selected=null;draft=state.board.map(row=>row.slice());$('edit-side').value=state.side;$('editor').hidden=false;document.body.classList.add('editor-active');renderPalette();render();
}
function endEditor(){editing=false;draft=null;$('editor').hidden=true;document.body.classList.remove('editor-active');render();}
function renderPalette(){
  $('palette').replaceChildren();
  for(const p of [...PIECES,'']){const b=document.createElement('button');b.textContent=p?pieceSymbol(p):'×';b.className=(p&&p===p.toUpperCase()?'red-piece ':'')+(p===palette?'selected':'');b.title=p?t(p===p.toUpperCase()?'Red':'Black')+' '+pieceName(p):t('Eraser');b.setAttribute('aria-label',b.title);b.onclick=()=>{palette=p;renderPalette();};$('palette').appendChild(b);}
}
function draftFen(){return draft.map(row=>{let out='',n=0;for(const p of row){if(!p)n++;else{if(n)out+=n;n=0;out+=p;}}if(n)out+=n;return out;}).join('/')+($('edit-side').value==='red'?' w':' b')+' - - 0 1';}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function modal(id){$(id).showModal();}
$('play').onclick=()=>{if(running){stop();return;}if(!busy){running=true;schedule();}};
$('step').onclick=()=>{stop();engineStep();};
$('undo').onclick=()=>{stop();action('undo');};
$('reset').onclick=()=>resetTo(state.initial_fen,title,presetId);
$('flip').onclick=()=>{flipped=!flipped;drawBoard();};
$('load-fen').onclick=()=>resetTo($('fen-input').value,'Your board',null);
$('copy-fen').onclick=async()=>{try{await navigator.clipboard.writeText(state.fen);notify(t("Board code copied."));}catch{$('fen-input').select();notify(t("Press Ctrl+C / Cmd+C to copy the board code."));}};
$('edit-toggle').onclick=startEditor;$('edit-cancel').onclick=endEditor;
$('clear-board').onclick=()=>{draft=Array.from({length:10},()=>Array(9).fill(''));drawBoard();};
$('edit-apply').onclick=async()=>{const fen=draftFen();if(await resetTo(fen,'Your board',null))endEditor();};
$('export').onclick=async()=>{try{const episode=await api('export');episode.controllers={red:control('red'),black:control('black')};download('xiangqi-episode-'+new Date().toISOString().replace(/[:.]/g,'-')+'.json',JSON.stringify(episode,null,2),'application/json');notify(t("Game saved."));}catch(e){notify(e.message,true);}};
$('import').onclick=()=>$('import-file').click();
$('import-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;stop();try{if(file.size>1000000)throw Error(t("The file must be smaller than 1 MB."));const episode=JSON.parse(await file.text());if(await action('import',{episode})){title='Loaded game';presetId=null;$('ply-limit').value=state.max_plies;render();notify(t("Game loaded."));}}catch(err){notify(err.message,true);}e.target.value='';};
for(const side of ['red','black'])$(side+'-model').onchange=()=>{stop();selected=null;render();};
$('guide-open').onclick=()=>modal('guide-dialog');
$('rules-open').onclick=()=>modal('rules-dialog');
$('history-panel').ontoggle=()=>{if($('history-panel').open)$('moves').scrollTop=$('moves').scrollHeight;};
document.querySelectorAll('[data-close]').forEach(el=>el.onclick=()=>$(el.dataset.close).close());
window.addEventListener('beforeunload',()=>{clearTimeout(timer);});
async function init(){
  try{
    const response=await fetch('/api/meta');meta=await response.json();if(!response.ok)throw Error(meta.error);
    // Keep traditional board markings independent of the menu language.
    // Removing the optional locale also removes its board symbols.
    if((meta.languages||[]).includes('zh-CN')){
      try{boardSymbols=await loadLocale('zh-CN');}catch{}
    }
    $('language').replaceChildren();
    for(const value of meta.languages||['en']){const option=document.createElement('option');option.value=value;option.textContent=value==='en'?'English':'Chinese';$('language').appendChild(option);}
    $('language').parentElement.hidden=(meta.languages||['en']).length<2;
    let preferred='en';try{preferred=localStorage.getItem('xiangqi-language')||'en';}catch{}
    await loadLanguage(preferred);
    const result=await api('session');session=result.session;state=result.state;render();
    $('connection').textContent=t("Local");
    if(!meta.engine_available){$('engine-health').textContent=t("Pikafish is not available");notify(t("Pikafish is missing. You can still use Player or Random."),true);}
  }catch(e){$('connection').textContent=t("Connection failed");notify(e.message+t(" Keep the local server window running."),true);document.querySelectorAll('button').forEach(b=>b.disabled=true);}
}
$('language').onchange=()=>loadLanguage($('language').value);
$('reload-models').onclick=async()=>{if(busy)return;stop();setBusy(true);try{const result=await api('models/reload');meta.models=result.models;modelCheck={message:'Model list updated.'};populateModels();renderModelList();render();}catch(e){notify(e.message,true);}finally{setBusy(false);}};
init();
