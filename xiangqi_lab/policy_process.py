"""One persistent, serialized JSON worker per Python model configuration."""
import atexit
from collections import deque
import json
import os
import queue
import subprocess
import threading
import uuid

class PolicyProcess:
    def __init__(self,command,cwd):
        self.lock=threading.RLock()
        self.output=queue.Queue()
        self.errors=deque(maxlen=12)
        self.process=subprocess.Popen(command,cwd=cwd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self.readers=[threading.Thread(target=self._read_out,daemon=True),threading.Thread(target=self._read_err,daemon=True)]
        for reader in self.readers:reader.start()

    def _read_out(self):
        for line in self.process.stdout:self.output.put(line)
        self.output.put(None)

    def _read_err(self):
        for line in self.process.stderr:self.errors.append(line[-1000:])

    def request(self,payload,timeout):
        with self.lock:
            request_id=uuid.uuid4().hex
            try:
                self.process.stdin.write(json.dumps({**payload,'request_id':request_id})+'\n');self.process.stdin.flush()
                response=self.output.get(timeout=timeout)
                if response is None:
                    self.readers[1].join(timeout=.2)
                    raise RuntimeError('Model exited unexpectedly: '+''.join(self.errors)[-1200:])
                result=json.loads(response)
                if not isinstance(result,dict) or result.get('request_id')!=request_id:
                    raise ValueError('Invalid model response or mismatched request ID.')
                if result.get('error'):raise RuntimeError('Model inference failed: '+str(result['error']))
                return result
            except queue.Empty:
                self.close()
                raise RuntimeError('Model timed out. Check its environment or increase timeout_seconds.')
            except OSError as exc:
                self.readers[1].join(timeout=.3)
                details=''.join(self.errors)[-1200:] or str(exc)
                self.close()
                raise RuntimeError('Model exited unexpectedly: '+details) from exc
            except Exception:
                self.close()
                raise

    def close(self):
        with self.lock:
            if self.process.poll() is None:self.process.kill()
            self.process.wait(timeout=3)
            for reader in self.readers:reader.join(timeout=1)
            for stream in (self.process.stdin,self.process.stdout,self.process.stderr):
                try:stream.close()
                except OSError:pass

POOL={}
POOL_LOCK=threading.Lock()

def get_worker(key,command,cwd):
    with POOL_LOCK:
        worker=POOL.get(key)
        if worker is None or worker.process.poll() is not None:
            worker=PolicyProcess(command,cwd);POOL[key]=worker
        return worker

def close_workers():
    with POOL_LOCK:
        for worker in POOL.values():worker.close()
        POOL.clear()

atexit.register(close_workers)
