"""Collect compact evidence and summarize the Orin precision comparison."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

out = Path('outputs/orin_c0_c2_fp16_int8')
out.mkdir(exist_ok=True, parents=True)
remote = '/home/q/Documents/Detr_Trials/benchmarks/20261026_c0_c2_fp16_int8'
def fetch(relative):
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes','q','cat '+remote+'/'+relative], capture_output=True, check=True)
    dest = out / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(result.stdout)
    return dest
for name in ['status.json','pipeline_status.json','restore.log','restore_verification.json','environment.txt','calibration_images.json','clocks.before.conf','source_identity.json']:
    try:
        fetch(name)
    except subprocess.CalledProcessError:
        pass
rows, evidence = [], {}
for model in ['c0','c2']:
    fetch(model+'/model_info.json')
    for precision in ['fp16','int8']:
        key = model+'/'+precision
        build = json.loads(fetch(key+'/build_info.json').read_text())
        accuracy = json.loads(fetch(key+'/accuracy.json').read_text())
        repeats, gpu, host = [], [], []
        for i in range(1,4):
            base = key+'/repeat'+str(i)
            info = json.loads(fetch(base+'/result.json').read_text())
            assert info['status'] == 'ok' and info['clocks_locked_and_unchanged']
            assert info['engine_sha256'] == build['engine_sha256']
            assert [b['shape'] for b in info['bindings'] if b['input']] == [[1,3,640,640]]
            assert info['warmup_iterations'] == 500 and info['requested_duration_s'] == 30
            assert info['measured_duration_s'] >= 30 and info['iterations'] > 0
            assert all(x['finite'] for x in info['output_checks'])
            repeats.append(info)
            values = list(csv.DictReader(fetch(base+'/latency.csv').open()))
            gpu += [float(x['gpu_compute_ms']) for x in values]
            host += [float(x['host_latency_ms']) for x in values]
        fetch(key+'/repeat1/engine_layers.json')
        fetch(key+'/build.log')
        gpu_sorted = sorted(gpu)
        def percentile(p):
            v=(len(gpu_sorted)-1)*p
            i=int(v)
            return gpu_sorted[i]+(gpu_sorted[min(i+1,len(gpu_sorted)-1)]-gpu_sorted[i])*(v-i)
        means = [r['gpu_compute']['mean_ms'] for r in repeats]
        row = {'model':model.upper(), 'precision':precision.upper(),
               'gpu_mean_ms':statistics.mean(gpu), 'gpu_p50_ms':percentile(.5),
               'gpu_p95_ms':percentile(.95), 'gpu_p99_ms':percentile(.99),
               'repeat_mean_min_ms':min(means), 'repeat_mean_max_ms':max(means),
               'host_mean_ms':statistics.mean(host),
               'serial_throughput_fps':sum(r['iterations'] for r in repeats)/sum(r['measured_duration_s'] for r in repeats),
               'AP_percent':accuracy['metrics_percent']['AP'],
               'AP50_percent':accuracy['metrics_percent']['AP50'],
               'APs_percent':accuracy['metrics_percent']['APs'],
               'engine_MiB':build['engine_bytes']/2**20,
               'mean_power_W':statistics.mean(r['resources']['mean_input_power_w'] for r in repeats if r['resources']['mean_input_power_w'] is not None) if any(r['resources']['mean_input_power_w'] is not None for r in repeats) else None,
               'samples':len(gpu)}
        rows.append(row)
        evidence[key] = {'build':build, 'accuracy':accuracy, 'repeats':repeats}
with (out/'summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(out/'summary.json').write_text(json.dumps({'rows':rows,'evidence':evidence},indent=2))
print(json.dumps(rows,indent=2))
