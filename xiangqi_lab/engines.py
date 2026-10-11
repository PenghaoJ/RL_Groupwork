"""UCI engine bridge and bounded JSON-lines model adapter."""
from pathlib import Path
import json
import os
import platform
import queue
import random
import re
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parent
CREATE_FLAGS=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

def uci_fen(fen):
    fields=fen.split()
    fields[0]=fields[0].translate(str.maketrans('HEhe','NBnb'))
    return ' '.join(fields)

def engine_path():
    name={"Windows":"Pikafish-Windows-x86-64-universal.exe","Darwin":"Pikafish-MacOS-universal","Linux":"Pikafish-Linux-x86-64-universal"}.get(platform.system(),"")
    return ROOT / "engine" / name

class UCIEngine:
    def __init__(self, command=None, cwd=None):
        path=engine_path()
        if command is None and not path.is_file(): raise RuntimeError("Pikafish is missing for this platform. Use the matching release package.")
        self.lines=queue.Queue()
        self.process=subprocess.Popen(command or [str(path)],cwd=cwd or path.parent,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1,creationflags=CREATE_FLAGS)
        threading.Thread(target=self._reader,daemon=True).start()
        try:
            self.send("uci"); self.wait(lambda line: line == "uciok",10)
            self.send(f"setoption name Threads value {min(4,os.cpu_count() or 1)}")
            self.send("setoption name Hash value 128")
            self.send("isready"); self.wait(lambda line: line == "readyok",10)
        except Exception:
            self.close(); raise

    def _reader(self):
        for line in self.process.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def send(self,line):
        self.process.stdin.write(line+"\n"); self.process.stdin.flush()

    def wait(self,predicate,timeout):
        deadline=time.monotonic()+timeout; seen=[]
        while True:
            try: line=self.lines.get(timeout=max(.01,deadline-time.monotonic()))
            except queue.Empty: raise RuntimeError("Engine timed out. Reduce the search budget or check the model.")
            if line is None: raise RuntimeError("Engine exited unexpectedly: "+"; ".join(seen[-4:]))
            seen.append(line)
            if predicate(line): return seen
            if time.monotonic() > deadline: raise RuntimeError("Engine response timed out.")

    def bestmove(self,game,milliseconds):
        self.send("ucinewgame")
        self.send("isready"); self.wait(lambda line: line == "readyok",10)
        position="position fen "+uci_fen(game.initial_fen)
        if game.history: position+=" moves "+" ".join(h["move"] for h in game.history)
        self.send(position)
        self.send(f"go movetime {milliseconds}")
        started=time.monotonic()
        lines=self.wait(lambda line: line.startswith("bestmove "),milliseconds/1000+12)
        move=lines[-1].split()[1]
        detail={"elapsed_ms":round((time.monotonic()-started)*1000),"side":game.side}
        for line in lines:
            if not line.startswith("info "): continue
            for field in ("depth","nodes","nps"):
                found=re.search(r"\b"+field+r" (\d+)",line)
                if found: detail[field]=int(found[1])
            score=re.search(r"score (cp|mate) (-?\d+)",line)
            if score:
                detail["score_type"]=score[1]
                detail["score_red"]=int(score[2])*(1 if game.side == "red" else -1)
            if " pv " in line: detail["pv"]=line.split(" pv ",1)[1].split()[:10]
        return move,detail

    def perft(self,fen,depth=1):
        self.send("position fen "+uci_fen(fen)); self.send(f"go perft {depth}")
        return self.wait(lambda line: line.startswith("Nodes searched:"),30)

    def close(self):
        try:
            if self.process.poll() is None:
                self.send("quit"); self.process.wait(timeout=2)
        except (OSError,subprocess.TimeoutExpired):
            self.process.kill(); self.process.wait(timeout=2)
        finally:
            for stream in (self.process.stdin,self.process.stdout):
                try:
                    if stream: stream.close()
                except OSError: pass

from model_registry import models as custom_models, catalog as model_catalog, command_for
from policy_process import get_worker, close_workers
from rules import action_move, Game, PRESETS


def choose_move(model, game, milliseconds=800):
    if game.outcome():
        raise ValueError('The game has ended.')
    started = time.monotonic()
    if model == 'random':
        move, detail = random.choice(game.legal), {'note': 'Untrained random baseline'}
    elif model == 'pikafish':
        engine = UCIEngine()
        try: move, detail = engine.bestmove(game, milliseconds)
        finally: engine.close()
    else:
        config = next((m for m in custom_models() if m['id'] == model), None)
        if not config: raise ValueError('Select a registered model.')
        command = command_for(config)
        if config['protocol'] == 'uci':
            engine = UCIEngine(command, cwd=config['_folder'])
            try: move, detail = engine.bestmove(game, milliseconds)
            finally: engine.close()
        else:
            payload = {'protocol': 'xiangqi-lab.agent.v1', 'observation': game.observation(), 'time_budget_ms': milliseconds}
            timeout = config.get('timeout_seconds', 30)
            if config['protocol'] == 'python':
                key = json.dumps(config, sort_keys=True)
                response = get_worker(key, command, config['_folder']).request(payload, timeout)
            else:
                try:
                    result = subprocess.run(command, input=json.dumps(payload)+'\n', text=True, encoding='utf-8', capture_output=True, cwd=config['_folder'], timeout=timeout, creationflags=CREATE_FLAGS)
                except subprocess.TimeoutExpired:
                    raise RuntimeError('Model timed out. Check its environment or increase timeout_seconds.')
                if result.returncode: raise RuntimeError('Model process failed: '+result.stderr[-1200:])
                try: response = json.loads(result.stdout.strip())
                except ValueError: raise ValueError('Model stdout must contain exactly one JSON response.')
            if not isinstance(response, dict): raise ValueError('Model response must be a JSON object.')
            if response.get('error'): raise RuntimeError('Model inference failed: '+str(response['error']))
            if ('move' in response) == ('action_id' in response):
                raise ValueError('Return exactly one of move or action_id.')
            move = action_move(response['action_id']) if 'action_id' in response else response['move']
            detail = {'note': 'Local model inference'}
    if not isinstance(move, str) or move not in game.legal:
        raise ValueError(f'Model returned an illegal move: {move!r}. The board was not changed.')
    detail['elapsed_ms'] = round((time.monotonic()-started)*1000)
    return move, detail


def check_model(model):
    # Independent positions: diagnostics never mutate a browser session.
    games = [Game(PRESETS[0]['fen']), Game(PRESETS[0]['fen']), Game(PRESETS[2]['fen'])]
    games[1].step(games[1].legal[0])
    results = []
    for game in games:
        move, details = choose_move(model, game, 100)
        results.append({'side': game.side, 'fen': game.fen, 'move': move, 'elapsed_ms': details['elapsed_ms']})
    return {'model': model, 'passed': True, 'cases': results, 'scope': 'Legal output on three sample positions; not a strength or universal compatibility guarantee.'}
