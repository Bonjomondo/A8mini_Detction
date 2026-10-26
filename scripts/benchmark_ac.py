#!/usr/bin/env python3
"""Build and benchmark local models on Orin NX (Test A + Test C only)."""
import argparse
import csv
import fcntl
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback

from benchmark_export import sha256
from benchmark_trt import clock_state, clocks_locked, distribution, resource_summary

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
TRTEXEC = '/usr/src/tensorrt/bin/trtexec'
CHINA = timezone(timedelta(hours=8))
MODELS = [
    ('B00_yolo11n', 'YOLO11n', 'yolo', 'yolo11n.pt'),
    ('B01_yolo11s', 'YOLO11s', 'yolo', 'yolo11s.pt'),
    ('B02_yolo11m', 'YOLO11m', 'yolo', 'yolo11m.pt'),
    ('B03_yolo26n', 'YOLO26n', 'yolo', 'yolo26n.pt'),
    ('B04_yolo26s', 'YOLO26s', 'yolo', 'yolo26s.pt'),
    ('B05_yolo26m', 'YOLO26m', 'yolo', 'yolo26m.pt'),
    ('B06_rtdetr_r18', 'RT-DETR-R18', 'rtdetr', 'rtdetr_r18vd_dec3_6x_coco_from_paddle.pth'),
    ('B07_sodetr_nofft', 'SO-DETR-NoFFT', 'nofft', 'so-detr-noFFT.pt'),
]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def command_output(command):
    try:
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=30)
        return {'command': command, 'exit_code': p.returncode, 'output': p.stdout.replace('\x00', '')}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'command': command, 'exit_code': None, 'output': str(e)}


def collect_environment(out, config):
    env_dir = out / 'env'
    env_dir.mkdir(exist_ok=True)
    commands = {
        'DATE': ['date', '--iso-8601=seconds'], 'MODEL': ['cat', '/proc/device-tree/model'],
        'KERNEL': ['uname', '-a'], 'OS': ['cat', '/etc/os-release'],
        'L4T': ['cat', '/etc/nv_tegra_release'],
        'CUDA': ['/usr/local/cuda/bin/nvcc', '--version'],
        'PACKAGES': ['dpkg-query', '-W', '-f=${Package}\t${Version}\n', 'nvidia-jetpack',
                     'nvidia-l4t-core', 'libcudnn8', 'libnvinfer8', 'cuda-toolkit-11-4'],
        'PYTHON_LIBRARIES': [sys.executable, '-c',
            "import json,platform,tensorrt,torch,ultralytics; print(json.dumps({"
            "'python':platform.python_version(),'tensorrt':tensorrt.__version__,"
            "'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),'cudnn':torch.backends.cudnn.version(),"
            "'ultralytics':ultralytics.__version__,'ultralytics_path':ultralytics.__file__}))"],
        'TRTEXEC_HELP': [TRTEXEC, '--help'],
        'NVP_MODEL': ['nvpmodel', '-q'],
        'NVP_MODEL_VERBOSE': ['sudo', '-n', 'nvpmodel', '-q', '--verbose'],
        'JETSON_CLOCKS': ['sudo', '-n', 'jetson_clocks', '--show'],
        'STORAGE': ['lsblk', '-o', 'NAME,SIZE,MODEL,TYPE,MOUNTPOINT'],
        'MEMORY': ['cat', '/proc/meminfo'],
    }
    results = {name: command_output(cmd) for name, cmd in commands.items()}
    # Optional authenticated readouts captured before starting this run.
    for name, filename in [('NVP_MODEL_VERBOSE', 'nvpmodel_verbose.txt'),
                           ('JETSON_CLOCKS', 'jetson_clocks_show.txt')]:
        dated = env_dir / (datetime.now(CHINA).strftime('%Y%m%d') +
                          ('_nvpmodel_verbose.txt' if name == 'NVP_MODEL_VERBOSE'
                           else '_jetson_clocks_show.txt'))
        recent = env_dir / ('nvpmodel_verbose_resume.txt' if name == 'NVP_MODEL_VERBOSE'
                            else 'jetson_clocks_resume.txt')
        saved = dated if dated.exists() else recent if recent.exists() else out / filename
        if saved.exists():
            results[name]['authenticated_snapshot'] = saved.read_text()
            results[name]['authenticated_snapshot_file'] = str(saved)
    data = {'captured_at': datetime.now(CHINA).isoformat(), 'commands': results,
            'clocks': clock_state(), 'benchmark': config,
            'carrier_board': 'DAMIAO DM-ORIN NX V2.X (user-provided; not inferred from device tree)'}
    stamp = datetime.now(CHINA).strftime('%Y%m%d_%H%M%S')
    snapshot = env_dir / ('environment_' + stamp + '.json')
    write_json(snapshot, data)
    if not (env_dir / 'environment.json').exists():
        write_json(env_dir / 'environment.json', data)
    raw = []
    for name, value in results.items():
        raw.extend(['===== {} ====='.format(name), 'Command: ' + ' '.join(value['command']),
                    'Exit code: ' + str(value['exit_code']), value['output']])
        if 'authenticated_snapshot' in value:
            raw.extend(['Authenticated snapshot:', value['authenticated_snapshot']])
    raw.append('===== CPU/GPU CLOCK SYSFS =====\n' + json.dumps(data['clocks'], indent=2))
    dump = env_dir / (datetime.now(CHINA).strftime('%Y%m%d') + '_orin_nx_env.txt')
    if not dump.exists():
        dump.write_text('\n'.join(raw))
    (env_dir / (stamp + '_orin_nx_env.txt')).write_text('\n'.join(raw))
    versions = {}
    for line in reversed(results['PYTHON_LIBRARIES']['output'].splitlines()):
        try:
            versions = json.loads(line)
            break
        except ValueError:
            pass
    packages = results['PACKAGES']['output']
    jetpack = next((line.split('\t', 1)[1] for line in packages.splitlines()
                    if line.startswith('nvidia-jetpack\t')), 'Unknown (nvidia-jetpack meta-package absent; see L4T)')
    if jetpack.startswith('Unknown') and 'REVISION: 6.4' in results['L4T']['output']:
        jetpack = '5.1.6 family (inferred from L4T 35.6.4; meta-package absent, actual compute versions recorded separately)'
    os_name = next((line.split('=', 1)[1].strip('"') for line in results['OS']['output'].splitlines()
                    if line.startswith('PRETTY_NAME=')), platform.platform())
    cudnn = versions.get('cudnn')
    cudnn_readable = '{}.{}.{}'.format(cudnn // 1000, cudnn % 1000 // 100, cudnn % 100) if isinstance(cudnn, int) else cudnn
    fan = []
    for f in sorted(Path('/sys/class/hwmon').glob('hwmon*')):
        for basename in ('name', 'rpm', 'pwm1'):
            try:
                fan.append('{}: {}'.format(f / basename, (f / basename).read_text().strip()))
            except OSError:
                pass
    md = '''# Orin NX benchmark environment

Hardware
--------------------------------
Platform: NVIDIA Jetson Orin NX
Memory: 16 GB advertised; /proc/meminfo saved in raw dump
Carrier Board: DAMIAO DM-ORIN NX V2.X (user-provided)
Storage: Samsung PM991a NVMe 128GB (lsblk saved)
Cooling: Active PWM fan detected; heatsink model not reported; dynamic fan control
Power Mode: {power}
jetson_clocks: {locked}; CPU/GPU sysfs checked before and after every repeat
Fixed CPU frequency (MHz): {cpu_mhz}
Fixed GPU frequency (MHz): {gpu_mhz}

Software
--------------------------------
OS: {os}
JetPack: {jetpack}
L4T: {l4t}
CUDA: {cuda}
cuDNN: {cudnn}
TensorRT: {trt}
Python: {python}
PyTorch: {torch}
Ultralytics: {ultralytics}; SO-DETR fork 8.0.201 for NoFFT

Benchmark
--------------------------------
Tests: A (pure TensorRT), C (concurrent resource monitoring)
Input Size: 640 × 640
Batch Size: 1
Precision: FP16 enabled; FP32 fallback allowed; TF32 disabled; FP32 input/output
Backend: TensorRT runtime execute_async_v2; CUDA events; no PyTorch in timing process
Dynamic Shape: No
Streams: 1; one inference in flight; synchronized serial loop
Warmup: {warmup} exact iterations in the same execution context, before each repeat
Benchmark: {duration} s per repeat
Repeat: {repeat} runs
Data Transfer: excluded (input uploaded once, outputs validated before timing)
Preprocessing: excluded (seed 0, device-resident random tensor)
Postprocessing: external detection selection/NMS excluded; model box decoding retained
CUDA Graph: Disabled
Builder Workspace: 4096 MiB
ONNX Opset: 16; static shape; no ONNX simplification
Test C: tegrastats every 200 ms; only samples inside the formal measurement window
Power: mean instantaneous VDD_IN readings; do not sum overlapping rails
RAM: whole-system RAM and increase over the process's pre-engine-load baseline
Temperature: maximum SoC sensor reading; Wi-Fi sensor excluded
Time zone: Asia/Shanghai (UTC+08:00)
Captured: {date}

Fan readout
--------------------------------
{fan}
'''.format(power=results['NVP_MODEL']['output'].split('NVPM ERROR')[0].strip().replace('\n', ' / '),
           locked='Enabled' if clocks_locked(data['clocks']) else 'Not verified',
           cpu_mhz=', '.join(str(v / 1000) for v in sorted({v for k, v in data['clocks'].items()
               if '/cpu' in k and k.endswith('/scaling_min_freq')})),
           gpu_mhz=', '.join(str(v / 1e6) for k, v in data['clocks'].items()
               if 'ga10b' in k and k.endswith('/min_freq')),
           os=os_name, jetpack=jetpack, l4t=results['L4T']['output'].strip(),
           cuda=versions.get('torch_cuda'), cudnn=cudnn_readable,
           trt=versions.get('tensorrt'), python=versions.get('python'), torch=versions.get('torch'),
           ultralytics=versions.get('ultralytics'), warmup=config['warmup_iterations'],
           duration=config['duration_s'], repeat=config['repeat'], date=data['captured_at'],
           fan='\n'.join(fan))
    (out / 'benchmark_env.md').write_text(md)
    if 'inferred' in jetpack:
        with open(out / 'benchmark_env.md', 'a') as f:
            f.write('\nJetPack family mapping source: [NVIDIA Jetson Linux 35.6.4](https://developer.nvidia.com/embedded/jetson-linux-r3564).\n')
    shutil.copyfile(out / 'benchmark_env.md', env_dir / ('benchmark_env_' + stamp + '.md'))
    data['snapshot_path'] = str(snapshot)
    return data


def aggregate(model_dir, config):
    import numpy as np
    runs = [json.loads((model_dir / 'run_{:02d}/result.json'.format(i)).read_text())
            for i in range(1, config['repeat'] + 1)]
    if any(r['status'] != 'ok' for r in runs):
        raise RuntimeError('Invalid repeat in {}'.format(model_dir))
    engine_hash = sha256(model_dir / 'model_fp16.engine')
    if any(r.get('engine_sha256', engine_hash) != engine_hash for r in runs):
        raise RuntimeError('Engine changed between repeats')
    gpu, host, telemetry = [], [], []
    for i, r in enumerate(runs, 1):
        run_dir = model_dir / 'run_{:02d}'.format(i)
        with open(run_dir / 'latency.csv') as f:
            for row in csv.DictReader(f):
                gpu.append(float(row['gpu_compute_ms']))
                host.append(float(row['host_latency_ms']))
        for line in (run_dir / 'telemetry.jsonl').read_text().splitlines():
            s = json.loads(line)
            if r['measurement_begin_monotonic_s'] <= s['monotonic_s'] <= r['measurement_end_monotonic_s']:
                telemetry.append(s)
    result = {'status': 'ok', 'config': config, 'model_info': json.loads((model_dir / 'model_info.json').read_text()),
              'engine_sha256': sha256(model_dir / 'model_fp16.engine'),
              'engine_bytes': (model_dir / 'model_fp16.engine').stat().st_size,
              'runs': runs, 'iterations': len(gpu), 'gpu_compute': distribution(gpu),
              'host_latency': distribution(host),
              'throughput_fps': sum(r['iterations'] for r in runs) / sum(r['measured_duration_s'] for r in runs),
              'run_mean_gpu_ms_std': statistics.stdev(r['gpu_compute']['mean_ms'] for r in runs) if len(runs) > 1 else 0,
              'resources': resource_summary(telemetry, None)}
    deltas = [r['resources']['peak_ram_above_baseline_mb'] for r in runs
              if r['resources']['peak_ram_above_baseline_mb'] is not None]
    result['resources']['peak_ram_above_baseline_mb'] = max(deltas) if deltas else None
    write_json(model_dir / 'result.json', result)
    return result


def report(out):
    results, rows, dates = [], [], {}
    for folder, model, kind, weights in MODELS:
        path = out / folder / 'result.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        results.append({'model': model, **r})
        row = {'model': model, 'status': r['status']}
        env_ref = r.get('config', {}).get('environment')
        if env_ref:
            snapshot = out / folder / env_ref
            if snapshot.exists():
                date = json.loads(snapshot.read_text())['captured_at'][:10]
                row['environment_date'] = date
                dates.setdefault(date, []).append(model)
        if r['status'] == 'ok':
            info, gpu, resource = r['model_info'], r['gpu_compute'], r['resources']
            row.update(params_m=info['deploy_params'] / 1e6, gflops=info.get('gflops'),
                       onnx_mib=info['onnx_bytes'] / 1024**2,
                       engine_mib=r['engine_bytes'] / 1024**2, mean_gpu_ms=gpu['mean_ms'],
                       run_mean_std_ms=r['run_mean_gpu_ms_std'], p50_ms=gpu['p50_ms'],
                       p95_ms=gpu['p95_ms'], p99_ms=gpu['p99_ms'],
                       mean_host_ms=r['host_latency']['mean_ms'], fps=r['throughput_fps'],
                       peak_system_ram_mb=resource['peak_system_ram_mb'],
                       ram_delta_mb=resource['peak_ram_above_baseline_mb'],
                       gpu_util_pct=resource['mean_gpu_util_pct'], power_w=resource['mean_input_power_w'],
                       peak_soc_c=resource['peak_soc_temp_c'])
        else:
            row['error'] = r.get('error')
        rows.append(row)
    write_json(out / 'summary.json', results)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with open(out / 'summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    def fmt(value, digits=2):
        return 'N/A' if value is None else format(value, '.{}f'.format(digits))
    config_path = out / 'config.json'
    settings = json.loads(config_path.read_text()) if config_path.exists() else {}
    md = ['# Orin NX Test A / Test C 实测结果', '',
          '统一 640×640、Batch 1、静态 TensorRT FP16；每次预热 {} 次，正式 {} 秒，重复 {} 次。电源模式、锁频和实际运行参数见 [benchmark_env.md](benchmark_env.md)。'.format(
              settings.get('warmup_iterations'), settings.get('duration_s'), settings.get('repeat')), '',
          'Test A 用 CUDA events 测 GPU compute，主机计时测 enqueue 到同步完成；单 stream、每次仅一个推理在执行。FPS 是完整串行计时循环的实测吞吐，包含必要的主机提交、事件同步和计时开销，不能用 1000/GPU time 替代。', '',
          '| Model | Params (M) | Estimated GFLOPs | GPU latency (ms) | Serial FPS | Peak system RAM (MiB) | Mean VDD_IN (W) | Peak SoC (°C) |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['status'] == 'ok':
            md.append('| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                row['model'], fmt(row['params_m']), fmt(row['gflops']),
                fmt(row['mean_gpu_ms'], 3), fmt(row['fps']),
                fmt(row['peak_system_ram_mb'], 0), fmt(row['power_w']), fmt(row['peak_soc_c'])))
        else:
            md.append('| {} | — | — | FAILED | — | — | — | — |'.format(row['model']))
    md += ['',
          '| Model | GPU mean ± run SD (ms) | P50 | P95 | P99 | Host latency (ms) | Serial FPS |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['status'] != 'ok':
            md.append('| {} | FAILED | — | — | — | — | — |'.format(row['model']))
            continue
        md.append('| {} | {} ± {} | {} | {} | {} | {} | {} |'.format(
            row['model'], fmt(row['mean_gpu_ms'], 3), fmt(row['run_mean_std_ms'], 3),
            fmt(row['p50_ms'], 3), fmt(row['p95_ms'], 3), fmt(row['p99_ms'], 3),
            fmt(row['mean_host_ms'], 3), fmt(row['fps'])))
    md += ['', 'Test C 与正式推理同时采样。RAM 为系统总占用，不等于模型独占显存；ΔRAM 为相对每次加载 engine 前的系统 RAM 的最大增加量。功耗为 VDD_IN 模块输入功率，不是 25W 模式名称，也不是载板电源插口的整机功率。', '',
           '| Model | Peak system RAM (MiB) | ΔRAM (MiB) | GPU util (%) | Mean VDD_IN (W) | Peak SoC temp (°C) |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['status'] == 'ok':
            md.append('| {} | {} | {} | {} | {} | {} |'.format(row['model'],
                fmt(row['peak_system_ram_mb'], 0), fmt(row['ram_delta_mb'], 0),
                fmt(row['gpu_util_pct'], 1), fmt(row['power_w']), fmt(row['peak_soc_c'], 2)))
    md += ['', '| Model | Deploy params (M) | Estimated GFLOPs | ONNX (MiB) | Engine (MiB) |',
           '|---|---:|---:|---:|---:|']
    for r in results:
        if r['status'] == 'ok':
            i = r['model_info']
            md.append('| {} | {} | {} | {} | {} |'.format(r['model'], fmt(i['deploy_params'] / 1e6),
                fmt(i.get('gflops')), fmt(i['onnx_bytes'] / 1024**2), fmt(r['engine_bytes'] / 1024**2)))
    md += ['', '统计口径：延迟分位数由 3 次全部逐次样本合并计算；SD 是 3 个 run mean 的样本标准差；功耗/利用率由全部正式 tegrastats 样本合并。THOP 估算 GFLOPs = MACs×2，不支持的算子未计入；部署参数量在融合/移除训练辅助分支后统计。', '',
           'YOLO26 保留训练好的 one-to-one 分支，仅排除最终 top-k 检测筛选；RT-DETR 官方权重 strict=True 加载，导出原始 boxes/scores；NoFFT 使用项目 SO_DETR fork。模型内置解码及 RT-DETR 的 query selection 属于网络图，保留计时。FP16 是 builder 开启的精度，TensorRT 允许必要的 FP32 回退；FP32 I/O，TF32 关闭。', '',
           '类别数：YOLO 和官方 RT-DETR 为 COCO 80 类，NoFFT 为 VisDrone 10 类。这是现有权重的部署性能基线，不用于隔离类别数影响或比较检测精度。', '',
           'RT-DETR 来源：[官方实现](https://github.com/lyuwenyu/RT-DETR)，提交号保存在该模型 model_info.json。', '',
           '每个模型目录保留 export.log、build.log、benchmark.log、config.json、model_info.json、result.json；每个 run 保留逐次 latency.csv、tegrastats.log、带单调时间戳的 telemetry.jsonl 和 result.json。未执行 Test B、AP、相机或 INT8。', '']
    if (out / 'archive').exists():
        md += ['本次恢复记录：初次导出的五个 YOLO ONNX 含有 NaN 卷积权重，已保存在 [archive](archive/)。重新导出使用 PyTorch 的逐通道 Conv/BN 融合；构建前检查 ONNX 权重有限值，每次预热后检查全部 engine 输出。YOLO11n 复用原有完整三轮结果，其余模型使用恢复后的环境快照；各模型 config.json 与各 run/command.json 可追溯实际快照。', '']
    if len(dates) > 1:
        md += ['跨日环境记录：' + '；'.join('{}（{}）'.format(date, '、'.join(names))
                   for date, names in sorted(dates.items())) + '。CPU/GPU 锁频条件一致；全系统 RAM 的绝对峰值还受桌面与后台进程影响，应结合各 run 的加载前 baseline_ram_mb 和 ΔRAM 判断资源占用。', '']
    pause_path = out / 'pause_state.json'
    if pause_path.exists():
        state = json.loads(pause_path.read_text())
        if state['status'] == 'paused':
            md[2:2] = ['按用户要求在当前模型完成后暂停；已完成 {}/{} 个模型。暂停时间：{}。待测：{}。续跑信息见 [pause_state.json](pause_state.json)。'.format(
                len(state['completed_models']), len(MODELS), state['paused_at'],
                '、'.join(state['pending_models'])), '']
    for row in rows:
        if row['status'] != 'ok':
            md.append('{} 失败：`{}`'.format(row['model'], row.get('error')))
    (out / 'summary.md').write_text('\n'.join(md))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'benchmarks' / datetime.now(CHINA).strftime('%Y%m%d_%H%M%S_testAC'))
    p.add_argument('--models', nargs='+', help='Subset, e.g. YOLO11n SO-DETR-NoFFT')
    p.add_argument('--duration', type=float, default=30)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--allow-unlocked', action='store_true')
    p.add_argument('--report-only', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(out)
        return
    # Hold this lock for the lifetime of the runner, including its GPU children.
    run_lock = open(out / 'runner.lock', 'a+')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('Another benchmark runner already owns ' + str(out))
    run_lock.seek(0)
    run_lock.truncate()
    run_lock.write(str(os.getpid()) + '\n')
    run_lock.flush()
    if args.models and any(m not in [v[1] for v in MODELS] for m in args.models):
        p.error('Unknown model; choices: ' + ', '.join(v[1] for v in MODELS))
    if args.duration <= 0 or args.warmup < 0 or args.repeat < 1:
        p.error('duration > 0, warmup >= 0, repeat >= 1 required')
    config = {'input_size': [640, 640], 'batch': 1, 'precision': 'FP16 with FP32 fallback',
              'backend': 'TensorRT', 'dynamic': False, 'streams': 1, 'cuda_graph': False,
              'warmup_iterations': args.warmup, 'duration_s': args.duration, 'repeat': args.repeat,
              'data_transfer': 'excluded', 'preprocessing': 'excluded',
              'postprocessing': 'external selection/NMS excluded; model box decode retained',
              'workspace_mib': 4096, 'opset': 16, 'tf32': False, 'seed': 0,
              'tests': ['A', 'C'], 'allow_unlocked': args.allow_unlocked}
    if (out / 'config.json').exists():
        previous_config = json.loads((out / 'config.json').read_text())
        if previous_config != config:
            p.error('Benchmark settings changed; use a new output directory')
    write_json(out / 'config.json', config)
    pause_path = out / 'pause_state.json'
    if pause_path.exists():
        state = json.loads(pause_path.read_text())
        if state['status'] == 'paused':
            state.update(status='resumed', resumed_at=datetime.now(CHINA).isoformat())
            write_json(pause_path, state)
    environment = collect_environment(out, config)
    snapshot = Path(environment['snapshot_path'])
    env_hash = sha256(snapshot)
    selected = [m for m in MODELS if not args.models or m[1] in args.models]
    for folder, name, kind, filename in selected:
        model_dir = out / folder
        model_dir.mkdir(exist_ok=True)
        weights = ROOT / 'weights' / filename
        if kind == 'nofft' and not weights.exists():
            candidates = list((ROOT / 'weights').glob('*noFFT*.pt')) + list((ROOT / 'weights').glob('*noFFN*.pt'))
            if len(candidates) == 1:
                weights = candidates[0]
        model_config = {**config, 'model': name, 'weights': str(weights),
                        'environment': '../env/' + snapshot.name, 'environment_sha256': env_hash}
        print('{}: preparing'.format(name), flush=True)
        try:
            if not weights.exists():
                raise FileNotFoundError(weights)
            if not (model_dir / 'model_info.json').exists():
                with open(model_dir / 'export.log', 'w') as log:
                    subprocess.run([sys.executable, str(SCRIPT_DIR / 'benchmark_export.py'), '--kind', kind,
                                    '--weights', str(weights), '--output', str(model_dir)],
                                   stdout=log, stderr=subprocess.STDOUT, check=True, cwd=SCRIPT_DIR)
            info = json.loads((model_dir / 'model_info.json').read_text())
            if info['weights_sha256'] != sha256(weights):
                raise RuntimeError('Weight changed since export; choose a new output directory')
            if info['onnx_sha256'] != sha256(model_dir / 'model.onnx'):
                raise RuntimeError('ONNX changed since export')
            import onnx
            import numpy as np
            for tensor in onnx.load(str(model_dir / 'model.onnx')).graph.initializer:
                values = onnx.numpy_helper.to_array(tensor)
                if values.dtype.kind == 'f' and not np.isfinite(values).all():
                    raise RuntimeError('Non-finite ONNX weight: ' + tensor.name)
            engine = model_dir / 'model_fp16.engine'
            old_build = model_dir / 'build_command.json'
            if engine.exists() and old_build.exists():
                recorded = json.loads(old_build.read_text())
                if recorded['engine_sha256'] != sha256(engine):
                    raise RuntimeError('Engine changed since build manifest')
            previous_result = model_dir / 'result.json'
            if previous_result.exists():
                completed = json.loads(previous_result.read_text())
                if completed['status'] == 'ok':
                    if any(completed['config'].get(k) != v for k, v in config.items()):
                        raise RuntimeError('Completed model settings changed')
                    if not engine.exists() or completed['engine_sha256'] != sha256(engine):
                        raise RuntimeError('Completed model engine changed')
                    if len(completed['runs']) != args.repeat:
                        raise RuntimeError('Completed model repeat count changed')
                    if any(r['clock_state_before'] != environment['clocks'] for r in completed['runs']):
                        raise RuntimeError('Completed model used different CPU/GPU frequency caps')
                    print('{}: all {} repeats already complete; preserving original evidence'.format(name, args.repeat), flush=True)
                    continue
            write_json(model_dir / 'config.json', model_config)
            build = [TRTEXEC, '--onnx=' + str(model_dir / 'model.onnx'), '--saveEngine=' + str(engine),
                     '--fp16', '--noTF32', '--workspace=4096', '--profilingVerbosity=detailed',
                     '--timingCacheFile=' + str(out / 'timing.cache'), '--buildOnly']
            if not engine.exists():
                print('{}: building TensorRT FP16'.format(name), flush=True)
                pending_engine = model_dir / 'model_fp16.building.engine'
                build[2] = '--saveEngine=' + str(pending_engine)
                with open(model_dir / 'build.log', 'w') as log:
                    subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True)
                pending_engine.replace(engine)
            write_json(model_dir / 'build_command.json', {'command': build,
                       'engine_sha256': sha256(engine), 'onnx_sha256': info['onnx_sha256']})
            if not clocks_locked(clock_state()) and not args.allow_unlocked:
                raise RuntimeError('Clocks not locked; run sudo jetson_clocks')
            if clock_state() != environment['clocks']:
                raise RuntimeError('Clock caps changed since environment capture')
            for repeat in range(1, args.repeat + 1):
                run_dir = model_dir / 'run_{:02d}'.format(repeat)
                run_dir.mkdir(exist_ok=True)
                existing = run_dir / 'result.json'
                if existing.exists():
                    old = json.loads(existing.read_text())
                    if old['status'] == 'ok' and old['warmup_iterations'] == args.warmup and old['requested_duration_s'] == args.duration:
                        if old.get('engine_sha256') != sha256(engine):
                            raise RuntimeError('Completed repeat used a different engine')
                        if old['clock_state_before'] != environment['clocks']:
                            raise RuntimeError('Completed repeat used different CPU/GPU frequency caps')
                        print('{}: repeat {} already complete'.format(name, repeat), flush=True)
                        continue
                    raise RuntimeError('Repeat settings changed; use a new output directory')
                print('{}: repeat {}/{} (warmup {}, benchmark {}s)'.format(
                    name, repeat, args.repeat, args.warmup, args.duration), flush=True)
                cmd = [sys.executable, str(SCRIPT_DIR / 'benchmark_trt.py'), '--engine', str(engine),
                       '--output', str(run_dir), '--warmup', str(args.warmup),
                       '--duration', str(args.duration)]
                if args.allow_unlocked:
                    cmd.append('--allow-unlocked')
                write_json(run_dir / 'command.json', {'command': cmd, **model_config})
                with open(run_dir / 'benchmark.log', 'w') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            with open(model_dir / 'benchmark.log', 'w') as log:
                for i in range(1, args.repeat + 1):
                    log.write('===== Repeat {} =====\n'.format(i))
                    log.write((model_dir / 'run_{:02d}/benchmark.log'.format(i)).read_text())
            result = aggregate(model_dir, model_config)
            power = result['resources']['mean_input_power_w']
            print('{}: {:.3f} ms GPU, {:.2f} serial FPS, {} W'.format(
                name, result['gpu_compute']['mean_ms'], result['throughput_fps'],
                'N/A' if power is None else '{:.2f}'.format(power)), flush=True)
        except Exception as e:
            write_json(model_dir / 'result.json', {'status': 'failed', 'config': model_config,
                                                 'error': str(e), 'traceback': traceback.format_exc()})
            print('{}: FAILED: {}'.format(name, e), flush=True)
        report(out)
    write_json(out / 'clock_state_final.json', clock_state())
    rows = report(out)
    failed = [r['model'] for r in rows if r['status'] != 'ok']
    print('Finished. Report: {}'.format(out / 'summary.md'), flush=True)
    if failed:
        print('Failed: ' + ', '.join(failed), flush=True)
        sys.exit(1)
    if len(rows) == len(MODELS):
        print('Verifying all recorded runs and environment snapshots', flush=True)
        with open(out / 'audit.log', 'w') as log:
            subprocess.run([sys.executable, str(SCRIPT_DIR / 'benchmark_audit.py'),
                            '--output', str(out)], stdout=log, stderr=subprocess.STDOUT, check=True)
        if pause_path.exists():
            state = json.loads(pause_path.read_text())
            state.update(status='complete', completed_at=datetime.now(CHINA).isoformat(),
                         completed_models=[r['model'] for r in rows], pending_models=[])
            write_json(pause_path, state)
        print('All 8 models complete; 24 runs verified', flush=True)


if __name__ == '__main__':
    main()
