"""Publish pilot progress and run all three one-loop settings after reasoning4."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pilot_state import ROOT,write_json,install_signal_handlers,pause_requested
from dory_pilot_progress import BASE,SPACES,build,publish


def main():
    install_signal_handlers()
    own=(BASE/'one_loop_queue.lock').open('a+')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(BASE/'supervisor.lock').open('a+')
    path=BASE/'one_loop_queue.json'
    state=dict(pid=os.getpid(),phase='waiting_for_reasoning4',order=[f'{s}1' for s in SPACES],publication_errors=[])
    child=None;last_publish=time.monotonic();last_build=0
    def signature():
        return tuple((BASE/f'{s}_loops{n}/generation_complete.json').exists() for s in SPACES for n in [1,2,4])
    previous=signature()
    def update(force=False):
        nonlocal last_publish,last_build,previous
        now=time.monotonic();sig=signature()
        if force or sig!=previous or now-last_publish>=900:
            try:
                build();publish();state['last_push_unix']=time.time();state.pop('last_publication_error',None)
            except Exception as exc:
                state['last_publication_error']=str(exc)
                state['publication_errors'].append(dict(time_unix=time.time(),error=str(exc)))
                print('Publication failed; generation continues:',exc,flush=True)
            last_publish=now;last_build=now;previous=sig
        elif now-last_build>=120:
            build();last_build=now
        state['updated_at_unix']=time.time();write_json(path,state)
    try:
        while True:
            update()
            if pause_requested(BASE):state['phase']='paused';return
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(10)
        if not all((BASE/f'{s}_loops{n}/generation_complete.json').exists() for s in SPACES for n in [2,4]):
            state['phase']='blocked_prior_settings_incomplete';return
        update(force=True)
        for space in SPACES:
            if pause_requested(BASE):state['phase']='paused';return
            out=BASE/f'{space}_loops1';out.mkdir(exist_ok=True)
            if (out/'generation_complete.json').exists():continue
            state.update(phase='generating',space=space,loops=1)
            write_json(path,state)
            with (out/'run.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_dory_large_pilot.py'),'worker',
                    '--space',space,'--loops','1'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['generation_pid']=child.pid;write_json(path,state)
                while child.poll() is None:
                    if pause_requested(BASE):child.terminate();child.wait(timeout=180);state['phase']='paused';return
                    update();time.sleep(10)
                if child.returncode or not (out/'generation_complete.json').exists():
                    state.update(phase='paused' if child.returncode==75 else 'failed',returncode=child.returncode);return
            state['generation_pid']=None;update(force=True)
        state['phase']='complete';update(force=True)
    finally:
        if child and child.poll() is None:child.terminate();child.wait(timeout=180)
        state['generation_pid']=None;write_json(path,state)
        build()

if __name__=='__main__':main()
