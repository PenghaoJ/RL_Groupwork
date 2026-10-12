"""Offline, loopback-only Xiangqi research workspace. Python standard library."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import argparse
import json
import mimetypes
import os
import socket
import threading
import time
import uuid
import webbrowser

from rules import Game, PRESETS
from engines import choose_move, model_catalog, engine_path, check_model, close_workers

ROOT=Path(__file__).resolve().parent
SESSIONS={}
SESSION_LOCK=threading.Lock()

class LocalHTTPServer(ThreadingHTTPServer):
    allow_reuse_address=False

    def server_bind(self):
        # Windows SO_REUSEADDR can let two servers silently share one port.
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

class Session:
    def __init__(self):
        self.game=Game(PRESETS[0]["fen"])
        self.revision=0
        self.lock=threading.Lock()
        self.touched=time.time()
    def snapshot(self):
        return {**self.game.snapshot(),"revision":self.revision}

class Handler(BaseHTTPRequestHandler):
    server_version="XiangqiLab/1.2.0"

    def send_json(self,status,data):
        body=json.dumps(data,ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store")
        self.end_headers(); self.wfile.write(body)

    def trusted(self):
        host=self.headers.get("Host","")
        allowed={f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}"}
        origin=self.headers.get("Origin")
        return host in allowed and (not origin or origin in {"http://"+h for h in allowed})

    def do_GET(self):
        if not self.trusted(): return self.send_json(403,{"error":"Only local same-origin access is allowed."})
        path=urlparse(self.path).path
        if path == "/api/meta":
            try: return self.send_json(200,{"presets":PRESETS,"models":model_catalog(),"engine_available":engine_path().is_file(),"version":"1.2.0","languages":["en"]+(["zh-CN"] if (ROOT/'static/locales/zh-CN.json').is_file() else []),"rules":"Research rules: threefold repetition / 120 non-capture plies; no perpetual-check/chase adjudication."})
            except ValueError as e: return self.send_json(400,{"error":str(e)})
        files={"/":"index.html","/app.js":"app.js","/style.css":"style.css","/favicon.svg":"favicon.svg"}
        for locale in (ROOT/'static'/'locales').glob('*.json'):
            files['/locales/'+locale.name]='locales/'+locale.name
        if path not in files: return self.send_json(404,{"error":"Page not found."})
        file=ROOT/"static"/files[path]
        body=file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",(mimetypes.guess_type(str(file))[0] or "application/octet-stream")+"; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Cache-Control","no-cache")
        self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        if not self.trusted(): return self.send_json(403,{"error":"Only local same-origin access is allowed."})
        if self.headers.get_content_type() != "application/json": return self.send_json(415,{"error":"Use a JSON request body."})
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0 < length <= 1048576: raise ValueError("Request size must be between 1 byte and 1 MB.")
            payload=json.loads(self.rfile.read(length))
            if not isinstance(payload,dict): raise ValueError("The request body must be a JSON object.")
            path=urlparse(self.path).path
            if path == '/api/models/reload':
                catalog=model_catalog()
                close_workers()
                return self.send_json(200,{'models':catalog})
            if path == '/api/models/check':
                return self.send_json(200,check_model(payload.get('model','')))
            if path == "/api/session":
                with SESSION_LOCK:
                    # Expire only unused sessions, never interrupt a running engine request.
                    for key in list(SESSIONS):
                        s=SESSIONS[key]
                        if time.time()-s.touched > 7200 and not s.lock.locked(): del SESSIONS[key]
                    if len(SESSIONS)>=64: raise ValueError("Too many experiment sessions. Restart the local service.")
                    sid=uuid.uuid4().hex; session=Session(); SESSIONS[sid]=session
                return self.send_json(200,{"session":sid,"state":session.snapshot()})
            session=SESSIONS.get(payload.get("session"))
            if not session: return self.send_json(404,{"error":"Session expired. Reload the page."})
            with session.lock:
                session.touched=time.time()
                if payload.get("revision") != session.revision:
                    return self.send_json(409,{"error":"The position changed. Please retry.","state":session.snapshot()})
                game=session.game
                if path == "/api/reset":
                    limit=payload.get("max_plies",160)
                    if type(limit) is not int or not 2 <= limit <= 1000: raise ValueError("The ply limit must be between 2 and 1000.")
                    session.game=Game(payload.get("fen",""),limit)
                elif path == "/api/move":
                    game.step(payload.get("move",""))
                elif path == "/api/undo": game.undo()
                elif path == "/api/engine":
                    if game.outcome(): raise ValueError("The game has ended.")
                    budget=payload.get("milliseconds",800)
                    if type(budget) is not int or not 100 <= budget <= 5000: raise ValueError("The search budget must be between 100 and 5000 milliseconds.")
                    model=payload.get("model","pikafish")
                    move,analysis=choose_move(model,game,budget)
                    if move not in game.legal: raise ValueError(f"Model returned an illegal move: {move!r}. The board was not changed.")
                    game.step(move,actor=model,analysis=analysis)
                elif path == "/api/export": return self.send_json(200,game.export())
                elif path == '/api/observation':
                    return self.send_json(200,{'observation':game.observation(),'revision':session.revision,'outcome':game.outcome()})
                elif path == "/api/import":
                    episode=payload.get("episode",{})
                    if not isinstance(episode,dict) or episode.get("schema") != "xiangqi-lab.episode.v1": raise ValueError("Unsupported episode format.")
                    moves=episode.get("moves",[])
                    if not isinstance(moves,list) or len(moves)>1000: raise ValueError("The episode is too long.")
                    limit=episode.get("max_plies",160)
                    if type(limit) is not int or not 2 <= limit <= 1000: raise ValueError("Invalid ply limit.")
                    imported=Game(episode.get("initial_fen",""),limit)
                    for entry in moves:
                        if not isinstance(entry,dict): raise ValueError("Invalid move record.")
                        imported.step(entry.get("move",""),actor=str(entry.get("actor","import"))[:100])
                    if episode.get("final_fen") and imported.fen != episode["final_fen"]: raise ValueError("The move history does not match the final FEN.")
                    session.game=imported
                else: return self.send_json(404,{"error":"API endpoint not found."})
                session.revision+=1
                return self.send_json(200,{"state":session.snapshot()})
        except (ValueError,TypeError,KeyError) as exc:
            return self.send_json(400,{"error":str(exc)})
        except Exception as exc:
            return self.send_json(500,{"error":str(exc)})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument("--no-browser",action="store_true")
    args=parser.parse_args()
    server=None
    for port in range(args.port,args.port+20):
        try: server=LocalHTTPServer(("127.0.0.1",port),Handler); break
        except OSError: continue
    if server is None: raise RuntimeError("No port is available. Stop an old server and try again.")
    server.daemon_threads=True
    url=f"http://127.0.0.1:{server.server_port}"
    print(f"Xiangqi Endgame Lab\n{url}\nKeep this window open. Press Ctrl+C to stop.",flush=True)
    if not args.no_browser: threading.Timer(.5,lambda: webbrowser.open(url)).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__ == "__main__": main()
