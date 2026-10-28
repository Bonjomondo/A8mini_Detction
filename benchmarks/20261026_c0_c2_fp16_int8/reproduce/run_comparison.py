import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', YOLO_AUTOINSTALL='false')
status = {'started': time.time(), 'completed': [], 'status': 'running'}
def save():
    (root / 'status.json').write_text(json.dumps(status, indent=2))
def run(stage, args, log):
    status['stage'] = stage
    save()
    print(stage, flush=True)
    with open(log, 'w') as f:
        subprocess.run(args, cwd=root, stdout=f, stderr=subprocess.STDOUT, check=True)
    status['completed'].append(stage)
    save()
try:
    images = sorted(Path('/home/q/Documents/Detr_Trials/dataset/VisDrone2019-DET-YOLO/images/train').glob('*.jpg'))
    random.Random(0).shuffle(images)
    assert len(images) >= 256
    records = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in images[:256]]
    (root / 'calibration_images.json').write_text(json.dumps(records, indent=2))
    env = subprocess.run(['sh', '-c', 'uname -a; cat /etc/nv_tegra_release; nvpmodel -q; cat /proc/device-tree/model; python3 -m pip list'], capture_output=True, text=True)
    (root / 'environment.txt').write_text(env.stdout + env.stderr)
    for model in ['c0', 'c2']:
        folder = root / model
        if not (folder / 'model_info.json').exists():
            run(model+'_export', [sys.executable, str(root / 'export_model.py'), '--root', str(root), '--model', model], folder / 'export.log')
        for precision in ['fp16', 'int8']:
            out = folder / precision
            out.mkdir(exist_ok=True)
            if not (out / 'build_info.json').exists():
                run(model+'_'+precision+'_build', [sys.executable, str(root / 'build_engine.py'), '--root', str(root), '--model', model, '--precision', precision], out / 'build.log')
    # Alternate precision order across repeats to expose temperature/order effects.
    for repeat in range(1, 4):
        combos = [('c0','fp16'),('c0','int8'),('c2','fp16'),('c2','int8')]
        if repeat % 2 == 0:
            combos.reverse()
        for model, precision in combos:
            out = root / model / precision
            dest = out / ('repeat'+str(repeat))
            if (dest / 'result.json').exists():
                continue
            run(model+'_'+precision+'_repeat'+str(repeat), [sys.executable, '/home/q/Documents/Detr_Trials/scripts/benchmark_trt.py', '--engine', str(out / 'model.engine'), '--output', str(dest), '--warmup', '500', '--duration', '30', '--seed', '0'], out / ('repeat'+str(repeat)+'.log'))
    status.update(status='complete', finished=time.time())
except Exception as e:
    status.update(status='failed', error=repr(e), finished=time.time())
    raise
finally:
    save()
