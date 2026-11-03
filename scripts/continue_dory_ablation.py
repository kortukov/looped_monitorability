"""Wait for the existing pilot supervisor, then insert the no-thinking ablation."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from pilot_state import ROOT,write_json,pause_requested,install_signal_handlers

BASE=ROOT/'results/dory_large_pilot25000'
HOLD='queue-output-no-thinking-ablation-before-reasoning\n'


def main():
    install_signal_handlers()
    state=dict(phase='waiting_for_output4',order=['output_no_thinking2','output_no_thinking4','reasoning2','reasoning4'])
    path=BASE/'ablation_queue.json'
    own=(BASE/'ablation_queue.lock').open('a+')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(path,state)
    lock=(BASE/'supervisor.lock').open('a+')
    while True:
        if pause_requested(BASE):state['phase']='paused';write_json(path,state);return
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(10)
    if not all((BASE/f'output_loops{n}/generation_complete.json').exists() for n in [2,4]):
        state['phase']='blocked_output_incomplete';write_json(path,state);return
    hold=BASE/'reasoning_loops2/PAUSE'
    if not hold.exists() or hold.read_text()!=HOLD:
        state['phase']='blocked_hold_changed';write_json(path,state);return
    hold.unlink()
    fcntl.flock(lock,fcntl.LOCK_UN)
    state['phase']='running_amended_sequence';write_json(path,state)
    child=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_dory_large_pilot.py'),'run'],cwd=ROOT)
    state.update(phase='supervisor_exited',returncode=child.returncode);write_json(path,state)

if __name__=='__main__':main()
