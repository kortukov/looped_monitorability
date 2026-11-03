"""Local supervisor: generate Dory 4→2→1, judging alongside generation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from pilot_state import ROOT,write_json,append_jsonl

STOP=False

def stopped(_s,_f):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-num-seqs-4',type=int,required=True)
    p.add_argument('--max-num-seqs-2',type=int,required=True)
    p.add_argument('--max-num-seqs-1',type=int,required=True)
    p.add_argument('--allow-full-run',action='store_true')
    args=p.parse_args()
    if not args.allow_full_run:p.error('Explicit full-evaluation authorization is required')
    os.chdir(ROOT)
    root=ROOT/'results';root.mkdir(exist_ok=True)
    lock=(root/'dory_full_sequence.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    infer=str(ROOT/'.venv-dory/bin/python');report_python=str(ROOT/'.venv-report/bin/python')
    settings={4:args.max_num_seqs_4,2:args.max_num_seqs_2,1:args.max_num_seqs_1}
    state=dict(pid=os.getpid(),started_at_unix=time.time(),order=[4,2,1],
               max_num_seqs=settings,phase='starting',generation_pid=None,judge_pids={},errors=[])
    state_path=root/'dory_full_sequence_state.json'
    judges={};children=[];handles=[];reports={};last_telemetry=0
    def launch(command,log):
        handle=log.open('a');handles.append(handle)
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
        children.append(child);return child
    def save():
        state['updated_at_unix']=time.time();write_json(state_path,state)
    def telemetry():
        nonlocal last_telemetry
        if time.monotonic()-last_telemetry<60:return
        result=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,memory.used,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True)
        append_jsonl(root/'dory_full_gpu_telemetry.jsonl',dict(time_unix=time.time(),gpu=result.stdout.strip()))
        last_telemetry=time.monotonic();save()
    def start_judge(n,out):
        command=[infer,'-u','scripts/gpqa_pilot.py','judge','--model','nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16',
                 '--loops',str(n),'--full-dataset','--output',str(out),'--watch','--judge-workers','8']
        judges[n]=launch(command,out/'judge.log');state['judge_pids'][str(n)]=judges[n].pid
    def publish_ready_reports():
        for number, judge in list(judges.items()):
            output=root/f'gpqa_full_dory_loops{number}_25000'
            if number not in reports and (output/'grading_complete.json').exists():
                reports[number]=launch([report_python,'scripts/report_dory_full.py','--results',str(output)],output/'report.log')
    try:
        for n in [4,2,1]:
            if STOP or (root/'PAUSE').exists():break
            out=root/f'gpqa_full_dory_loops{n}_25000';out.mkdir(exist_ok=True)
            state.update(current_loops=n,phase='generating');save()
            for attempt in range(3):
                if STOP:break
                command=[infer,'-u','scripts/generate_dory_full.py','--loops',str(n),
                         '--max-num-seqs',str(settings[n]),'--allow-full-run']
                gen=launch(command,out/'run.log');state['generation_pid']=gen.pid;save()
                while gen.poll() is None and not STOP:
                    if n not in judges and (out/'config.json').exists():start_judge(n,out);save()
                    publish_ready_reports();telemetry();time.sleep(2)
                if STOP:
                    gen.terminate();break
                if gen.returncode==0:break
                if gen.returncode==75:
                    state['phase']='paused'
                    save();return
                state['errors'].append(f'loops={n}: generation attempt {attempt+1} exited {gen.returncode}')
                save()
                if attempt==2:raise RuntimeError(f'Generation failed three times at {n} loops')
                time.sleep(10)
            if STOP:break
            if not (out/'generation_complete.json').exists():raise RuntimeError(f'Missing completion marker for {n} loops')
            state['generation_pid']=None;save()
        if not STOP:
            state['phase']='finishing_judgments';save()
            for n in [4,2,1]:
                out=root/f'gpqa_full_dory_loops{n}_25000'
                for attempt in range(3):
                    if n not in judges or (judges[n].poll() is not None and judges[n].returncode!=0):start_judge(n,out)
                    judge=judges[n]
                    while judge.poll() is None and not STOP:publish_ready_reports();telemetry();time.sleep(2)
                    if STOP:break
                    if judge.returncode==0 and (out/'grading_complete.json').exists():break
                    state['errors'].append(f'loops={n}: judging attempt {attempt+1} did not complete');save()
                    if attempt==2:raise RuntimeError(f'Judging incomplete for {n} loops; raw outputs preserved')
                    time.sleep(10)
                if STOP:break
                if n in reports:
                    reports[n].wait()
                if n not in reports or reports[n].returncode != 0:
                    subprocess.run([report_python,'scripts/report_dory_full.py','--results',str(out)],check=True)
            if not STOP:
                subprocess.run([report_python,'scripts/report_dory_full.py','--compare'],check=True)
                state['phase']='complete';state['completed_at_unix']=time.time();save()
                print('All three full Dory evaluations and reports are complete.',flush=True)
    except BaseException as exc:
        state['phase']='failed';state['errors'].append(str(exc));save();raise
    finally:
        if STOP:state['phase']='paused'
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            if child.poll() is None:
                try:child.wait(timeout=180)
                except subprocess.TimeoutExpired:child.kill()
        state['generation_pid']=None;save()
        for handle in handles:handle.close()

if __name__=='__main__':main()
