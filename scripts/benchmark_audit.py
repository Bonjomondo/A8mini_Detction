#!/usr/bin/env python3
"""Verify the recorded Test A/C experiment without running GPU inference."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from benchmark_ac import MODELS
from benchmark_export import sha256


def audit_model(base, folder, settings):
    model_dir = base / folder
    result = json.loads((model_dir / 'result.json').read_text())
    assert result['status'] == 'ok', result.get('error', result['status'])
    assert all(result['config'].get(k) == v for k, v in settings.items()), 'Configuration mismatch'
    info = result['model_info']
    engine_hash = sha256(model_dir / 'model_fp16.engine')
    assert engine_hash == result['engine_sha256'], 'Engine checksum mismatch'
    assert sha256(model_dir / 'model.onnx') == info['onnx_sha256'], 'ONNX checksum mismatch'
    assert sha256(Path(info['weights'])) == info['weights_sha256'], 'Weight checksum mismatch'
    build = json.loads((model_dir / 'build_command.json').read_text())
    assert build['engine_sha256'] == engine_hash, 'Build manifest mismatch'
    assert '--fp16' in build['command'] and '--noTF32' in build['command'], 'Builder precision mismatch'
    assert '--workspace=4096' in build['command'], 'Builder workspace mismatch'
    runs = []
    total_iterations = 0
    total_duration = 0
    gpu_sum = 0
    reference_clocks = None
    for i in range(1, settings['repeat'] + 1):
        run_dir = model_dir / ('run_%02d' % i)
        r = json.loads((run_dir / 'result.json').read_text())
        command = json.loads((run_dir / 'command.json').read_text())
        env_path = (model_dir / command['environment']).resolve()
        assert sha256(env_path) == command['environment_sha256'], 'Environment checksum mismatch'
        environment = json.loads(env_path.read_text())
        assert all(command.get(k) == v for k, v in settings.items()), 'Run command mismatch'
        assert r['status'] == 'ok', 'Invalid run status'
        if 'engine_sha256' in r:
            assert r['engine_sha256'] == engine_hash, 'Run engine checksum mismatch'
        assert r['warmup_iterations'] == settings['warmup_iterations'], 'Warmup count mismatch'
        assert r['requested_duration_s'] == settings['duration_s'], 'Duration setting mismatch'
        assert r['measured_duration_s'] >= settings['duration_s'], 'Measurement window too short'
        assert r['clocks_locked_and_unchanged'], 'Unlocked or changed frequencies'
        assert r['clock_state_before'] == environment['clocks'], 'Environment clock mismatch'
        if reference_clocks is None:
            reference_clocks = r['clock_state_before']
        assert r['clock_state_before'] == reference_clocks, 'Repeat frequency mismatch'
        assert r['output_checks'] and all(c['finite'] for c in r['output_checks']), 'Non-finite output'
        inputs = [b for b in r['bindings'] if b['input']]
        assert len(inputs) == 1 and inputs[0]['shape'] == [1, 3, 640, 640], 'Input shape mismatch'
        assert all(b['dtype'] == 'float32' for b in r['bindings']), 'I/O precision mismatch'
        with (run_dir / 'latency.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == r['iterations'] > 0, 'Latency sample count mismatch'
        times = [float(row['gpu_compute_ms']) for row in rows]
        assert all(math.isfinite(t) and t > 0 for t in times), 'Invalid GPU latency'
        assert math.isclose(sum(times) / len(times), r['gpu_compute']['mean_ms'], rel_tol=1e-9), 'Run mean mismatch'
        samples = [json.loads(line) for line in (run_dir / 'telemetry.jsonl').read_text().splitlines()]
        samples = [s for s in samples if r['measurement_begin_monotonic_s'] <= s['monotonic_s'] <= r['measurement_end_monotonic_s']]
        assert len(samples) == r['resources']['samples'] > 0, 'Telemetry window/count mismatch'
        assert r['resources']['mean_input_power_w'] is not None, 'Missing power samples'
        gpu_cap = next(v / 1e6 for k, v in reference_clocks.items()
                       if 'ga10b' in k and k.endswith('/min_freq'))
        assert all(s.get('gpu_devfreq_mhz') == gpu_cap for s in samples), 'Unexpected GPU frequency'
        gpu_sum += sum(times)
        total_iterations += len(times)
        total_duration += r['measured_duration_s']
        runs.append({'repeat': i, 'duration_s': r['measured_duration_s'], 'iterations': len(times),
                     'telemetry_samples': len(samples), 'environment': str(env_path.relative_to(base)),
                     'output_finite': True, 'clocks_verified': True,
                     'engine_checksum_scope': 'run + model + build' if 'engine_sha256' in r
                         else 'model + build; legacy run did not record its own checksum'})
    assert len(result['runs']) == settings['repeat'], 'Aggregate repeat count mismatch'
    assert total_iterations == result['iterations'], 'Aggregate iteration count mismatch'
    assert math.isclose(gpu_sum / total_iterations, result['gpu_compute']['mean_ms'], rel_tol=1e-9), 'Aggregate mean mismatch'
    assert math.isclose(total_iterations / total_duration, result['throughput_fps'], rel_tol=1e-9), 'Aggregate FPS mismatch'
    layers = json.loads((model_dir / 'run_01/engine_layers.json').read_text())['Layers']
    convolutions = [l for l in layers if 'Convolution' in l['LayerType']]
    fp16_convolutions = sum(l.get('Weights', {}).get('Type') == 'Half' for l in convolutions)
    assert fp16_convolutions > 0, 'No FP16 convolution found in engine inspector'
    return {'status': 'ok', 'engine_sha256': engine_hash, 'runs': runs,
            'convolution_layers': len(convolutions), 'fp16_convolution_layers': fp16_convolutions,
            'total_iterations': total_iterations, 'measured_duration_s': total_duration}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--models', nargs='+', choices=[m[1] for m in MODELS])
    args = p.parse_args()
    base = args.output.resolve()
    settings = json.loads((base / 'config.json').read_text())
    selected = [m for m in MODELS if not args.models or m[1] in args.models]
    results = []
    for folder, name, _, _ in selected:
        try:
            item = audit_model(base, folder, settings)
        except Exception as e:
            item = {'status': 'failed', 'error': str(e)}
        results.append({'model': name, **item})
    passed = sum(r['status'] == 'ok' for r in results)
    report = {'status': 'ok' if passed == len(selected) else 'failed',
              'captured_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'passed_models': passed, 'expected_models': len(selected),
              'requested_models': [m[1] for m in selected], 'models': results}
    (base / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
