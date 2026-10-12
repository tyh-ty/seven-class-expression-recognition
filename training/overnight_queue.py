"""Local sequential job runner; stays awake until the authorized deadline."""
from __future__ import annotations
import ctypes
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'training/overnight-20260927'
DEADLINE = datetime.fromisoformat('2026-09-28T12:00:00+08:00')

def save(path,data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)

def process_running(pid):
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_uint,ctypes.c_int,ctypes.c_uint]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        return False
    code = ctypes.c_uint()
    try:
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
            raise ctypes.WinError()
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)

def main():
    global WORK, DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--deadline', default=DEADLINE.isoformat())
    args = parser.parse_args()
    WORK = args.work.resolve()
    if not WORK.is_relative_to(ROOT):
        raise ValueError('Queue work directory must be within the project')
    DEADLINE = datetime.fromisoformat(args.deadline)
    if DEADLINE.tzinfo is None:
        raise ValueError('Deadline must include its timezone')
    WORK.mkdir(exist_ok=True)
    lock = WORK/'queue.lock'
    try:
        fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Queue already locked; inspect its PID before starting another runner')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    status_path = WORK/'queue-status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {'jobs':{}}
    current = None
    log_handles = []
    kernel = ctypes.windll.kernel32
    try:
        kernel.SetThreadExecutionState(0x80000001)  # Prevent idle sleep; no global power-setting change.
        while datetime.now(timezone.utc) < DEADLINE or current is not None:
            now = datetime.now(timezone.utc)
            status.update(pid=os.getpid(),updated_at=now.isoformat(),deadline=DEADLINE.isoformat())
            queue_path = WORK/'queue.json'
            queue = json.loads(queue_path.read_text(encoding='utf-8')) if queue_path.exists() else []
            if current is not None:
                job,process = current
                finished = not process_running(job['wait_pid']) if 'wait_pid' in job else process.poll() is not None
                if finished:
                    code = 0 if process is None else process.returncode
                    missing = [p for p in job.get('require_after',[]) if not (ROOT/p).exists()]
                    status['jobs'][job['id']].update(status='completed' if code==0 and not missing else 'failed',exit_code=code,missing=missing,finished_at=now.isoformat())
                    for handle in log_handles:handle.close()
                    log_handles=[];current=None
            if current is None and now < DEADLINE:
                for job in queue:
                    if job['id'] in status['jobs']:
                        continue
                    dependencies = [status['jobs'].get(i,{}).get('status') for i in job.get('depends_on',[])]
                    if any(x=='failed' or x=='skipped' for x in dependencies):
                        status['jobs'][job['id']]={'status':'skipped','reason':'dependency failed'}
                        continue
                    if any(x!='completed' for x in dependencies):
                        continue
                    if any(not (ROOT/p).exists() for p in job.get('require_before',[])):
                        continue
                    if 'wait_pid' in job:
                        process=None
                    else:
                        stdout=(WORK/(job['id']+'.log')).open('a',encoding='utf-8')
                        stderr=(WORK/(job['id']+'-error.log')).open('a',encoding='utf-8')
                        log_handles=[stdout,stderr]
                        command=[sys.executable,'-X','utf8','-u',*job['args']]
                        process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                    status['jobs'][job['id']]={'status':'running','pid':job.get('wait_pid',process.pid if process else None),'started_at':now.isoformat()}
                    current=(job,process)
                    print('started',job['id'],status['jobs'][job['id']]['pid'],flush=True)
                    break
            status['status']='running' if current else 'waiting_for_jobs'
            save(status_path,status)
            time.sleep(15)
        status['status']='deadline_reached';save(status_path,status)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        for handle in log_handles:handle.close()
        lock.unlink(missing_ok=True)

if __name__ == '__main__':
    main()
