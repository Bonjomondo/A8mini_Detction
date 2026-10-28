"""Run held-out evaluation after the GPU benchmark queue, never concurrently."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
state = {'stage':'waiting_for_benchmarks', 'status':'running', 'started':time.time()}
def save():
    (root/'pipeline_status.json').write_text(json.dumps(state,indent=2))
try:
    save()
    while True:
        status=json.loads((root/'status.json').read_text())
        if status['status'] != 'running':
            break
        time.sleep(5)
    if status['status'] != 'complete':
        raise RuntimeError('Benchmark queue failed: '+str(status))
    state['stage']='held_out_evaluation'
    save()
    with open(root/'evaluation.log','w') as log:
        subprocess.run([sys.executable,str(root/'evaluate_engines.py')],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=root)
    state.update(stage='finished',status='complete')
except Exception as e:
    state.update(status='failed',error=repr(e))
    raise
finally:
    state['finished']=time.time()
    save()
