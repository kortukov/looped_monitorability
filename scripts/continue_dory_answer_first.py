"""Reload reporting at a generation boundary, finish extensions, then answer-first pilots."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pilot_state import ROOT,write_json,install_signal_handlers,pause_requested
from dory_pilot_progress import BASE,build,publish


def process_running(pid):
    try:return (open(f'/proc/{pid}/stat').read().split(') ',1)[1].split()[0] not in ('Z','X'))
    except FileNotFoundError:return False


def main():
    install_signal_handlers()
    own=(BASE/'answer_first_handoff.lock').open('a+')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=BASE/'documentation_handoff.json';state=json.loads(path.read_text())
    parent=state['supervisor_pid'];worker=state['generation_pid'];resumed=False;child=None
    try:
        while process_running(worker):
            state['phase']='waiting_for_current_generation';state['updated_at_unix']=time.time();write_json(path,state)
            time.sleep(10)
        if process_running(parent):os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
        resumed=True
        lock=(BASE/'supervisor.lock').open('a+')
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(2)
        if not (ROOT/state['output']/'generation_complete.json').exists():
            state['phase']='blocked_current_generation_incomplete';return
        fcntl.flock(lock,fcntl.LOCK_UN)
        if pause_requested(BASE):state['phase']='paused';return
        for script,phase in [('run_dory_extensions.py','finishing_extensions'),('run_dory_answer_first.py','running_answer_first')]:
            state['phase']=phase;write_json(path,state)
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts'/script),'run'],cwd=ROOT)
            while child.poll() is None:
                if pause_requested(BASE):child.terminate();child.wait(timeout=200);state['phase']='paused';return
                time.sleep(5)
            if child.returncode:state.update(phase='failed',returncode=child.returncode);return
            completion_state=BASE/'extensions_state.json' if phase=='finishing_extensions' else BASE/'answer_first/state.json'
            if json.loads(completion_state.read_text())['phase']!='complete':
                state['phase']='blocked_'+phase;return
        state['phase']='complete'
    finally:
        if not resumed and process_running(parent):os.kill(parent,signal.SIGCONT)
        if child and child.poll() is None:child.terminate();child.wait(timeout=200)
        state['updated_at_unix']=time.time();write_json(path,state)

if __name__=='__main__':main()
