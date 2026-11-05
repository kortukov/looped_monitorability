"""Add paired seeds 43/44 for thinking-off at 4/8 loops after answer-first pilots."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from pilot_state import ROOT,write_json,install_signal_handlers,pause_requested
from dory_pilot_progress import BASE,build,publish
from run_dory_extensions import paths

TASKS=[(seed,n) for seed in [43,44] for n in [4,8]]


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0] not in ('Z','X')
    except FileNotFoundError:return False


def main(args):
    install_signal_handlers()
    own=(BASE/'high_loop_seeds.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),phase='waiting_for_reasoning8',tasks=TASKS,publication_errors=[])
    path=BASE/'high_loop_seeds_state.json';child=None;resumed=not args.held_supervisor
    last_build=0;last_push=0
    def update(force=False):
        nonlocal last_build,last_push
        now=time.monotonic()
        if force or now-last_push>=900:
            try:build();publish();state['last_push_unix']=time.time()
            except Exception as exc:state['publication_errors'].append(dict(time_unix=time.time(),error=str(exc)));print(exc,flush=True)
            last_build=now;last_push=now
        elif now-last_build>=120:build();last_build=now
        state['updated_at_unix']=time.time();write_json(path,state)
    try:
        if args.held_supervisor:
            # Hold only reporting; the existing GPU worker remains untouched.
            while alive(args.held_worker):
                if pause_requested(BASE):state['phase']='paused';return
                update();time.sleep(5)
            if alive(args.held_supervisor):os.kill(args.held_supervisor,signal.SIGCONT)
            resumed=True
        state['phase']='waiting_for_answer_first';write_json(path,state)
        while True:
            if pause_requested(BASE):state['phase']='paused';return
            marker=BASE/'answer_first/state.json'
            if marker.exists() and json.loads(marker.read_text()).get('phase')=='complete':break
            state['updated_at_unix']=time.time();write_json(path,state);time.sleep(10)
        lock=(BASE/'supervisor.lock').open('a+')
        while True:
            if pause_requested(BASE):state['phase']='paused';return
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(5)
        if not (BASE/'reasoning_loops8/generation_complete.json').exists():
            state['phase']='blocked_prior_generation_incomplete';return
        update(force=True)
        for seed,loops in TASKS:
            if pause_requested(BASE):state['phase']='paused';return
            _,out=paths('output_no_thinking',loops,seed);out.mkdir(parents=True,exist_ok=True)
            if (out/'generation_complete.json').exists():continue
            state.update(phase='generating',sampling_seed=seed,loops=loops)
            with (out/'run.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_dory_extensions.py'),'worker',
                    '--space','output_no_thinking','--loops',str(loops),'--sampling-seed',str(seed)],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['generation_pid']=child.pid;write_json(path,state)
                while child.poll() is None:
                    if pause_requested(BASE):child.terminate();child.wait(timeout=180);state['phase']='paused';return
                    update();time.sleep(5)
                if child.returncode or not (out/'generation_complete.json').exists():
                    state.update(phase='paused' if child.returncode==75 else 'failed',returncode=child.returncode);return
            state['generation_pid']=None;update(force=True)
        state['phase']='complete';update(force=True)
    finally:
        if not resumed and alive(args.held_supervisor):os.kill(args.held_supervisor,signal.SIGCONT)
        if child and child.poll() is None:child.terminate();child.wait(timeout=180)
        state['generation_pid']=None;write_json(path,state)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--held-supervisor',type=int);p.add_argument('--held-worker',type=int)
    a=p.parse_args()
    if bool(a.held_supervisor)!=bool(a.held_worker):p.error('Specify both held supervisor and worker')
    main(a)
