#!/usr/bin/env python3
"""Test A/C: serial TensorRT runtime timing and concurrent Jetson telemetry.

No PyTorch is imported in this process. Device buffers, transfers and output
validation are outside timing. CUDA events measure GPU time; the host timer
measures enqueue + completion synchronization, with one inference in flight.
"""
import argparse
import csv
import ctypes as C
import ctypes.util
import glob
import json
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

from benchmark_export import sha256


def clock_state():
    paths = glob.glob('/sys/devices/17000000.ga10b/devfreq/*/min_freq')
    paths += glob.glob('/sys/devices/17000000.ga10b/devfreq/*/max_freq')
    paths += glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_min_freq')
    paths += glob.glob('/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_max_freq')
    result = {}
    for path in paths:
        try:
            result[path] = int(Path(path).read_text().strip())
        except (OSError, ValueError):
            pass
    return result


def clocks_locked(state):
    mins = [p for p in state if p.endswith('min_freq')]
    return bool(mins) and all(state[p] == state.get(p.replace('min_freq', 'max_freq')) for p in mins)


def parse_tegrastats(line):
    data = {}
    ram = re.search(r'\bRAM (\d+)/(\d+)MB', line)
    if ram:
        data.update(ram_mb=int(ram[1]), ram_total_mb=int(ram[2]))
    gpu = re.search(r'GR3D_FREQ (\d+)%', line)
    if gpu:
        data['gpu_util_pct'] = int(gpu[1])
    freq = re.search(r'GR3D_FREQ \d+%\s*@\[?([\d,]+)', line)
    if freq:
        data['gpu_freq_mhz'] = [int(v) for v in freq[1].split(',')]
    temperatures = {k: float(v) for k, v in re.findall(r'([\w]+)@(-?[\d.]+)C', line)}
    data['temperatures_c'] = temperatures
    # Current rail values only. Values after '/' are the tool's running averages.
    data['power_mw'] = {k: int(v) for k, v in re.findall(r'\b(VDD_\w+|POM_\w+)\s+(\d+)(?:mW)?/\d+(?:mW)?', line)}
    return data


class Telemetry:
    def __init__(self, out):
        self.samples = []
        self.error = None
        self.out = out

    def start(self):
        self.raw = open(self.out / 'tegrastats.log', 'w')
        self.rows = open(self.out / 'telemetry.jsonl', 'w')
        try:
            self.proc = subprocess.Popen(['tegrastats', '--interval', '200'],
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1)
        except OSError as e:
            self.error = repr(e)
            self.proc = None
            return

        def collect():
            for line in self.proc.stdout:
                sample = {'monotonic_s': time.perf_counter(), 'unix_s': time.time(),
                          **parse_tegrastats(line)}
                # Some carrier images expose utilization to tegrastats but report
                # GR3D frequency as zero without root. Preserve raw output and also
                # read the unprivileged devfreq attribute for actual frequency.
                for path in glob.glob('/sys/devices/17000000.ga10b/devfreq/*/cur_freq'):
                    try:
                        sample['gpu_devfreq_mhz'] = int(Path(path).read_text().strip()) / 1e6
                    except (OSError, ValueError):
                        pass
                self.raw.write(line)
                self.raw.flush()
                self.rows.write(json.dumps(sample) + '\n')
                self.rows.flush()
                if 'ram_mb' in sample:
                    self.samples.append(sample)
        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.thread.join(timeout=3)
        self.raw.close()
        self.rows.close()


class CUDA:
    def __init__(self):
        candidates = ['/usr/local/cuda/lib64/libcudart.so', ctypes.util.find_library('cudart')]
        self.lib = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.lib = C.CDLL(candidate)
                break
            except OSError:
                pass
        if self.lib is None:
            raise RuntimeError('CUDA runtime library not found')
        signatures = {
            'cudaSetDevice': [C.c_int], 'cudaMalloc': [C.POINTER(C.c_void_p), C.c_size_t],
            'cudaFree': [C.c_void_p], 'cudaMemcpy': [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
            'cudaStreamCreate': [C.POINTER(C.c_void_p)], 'cudaStreamDestroy': [C.c_void_p],
            'cudaStreamSynchronize': [C.c_void_p], 'cudaEventCreate': [C.POINTER(C.c_void_p)],
            'cudaEventDestroy': [C.c_void_p], 'cudaEventRecord': [C.c_void_p, C.c_void_p],
            'cudaEventSynchronize': [C.c_void_p],
            'cudaEventElapsedTime': [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
        }
        for name, types in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = types
            fn.restype = C.c_int
        self.call('cudaSetDevice', 0)

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError('{} failed with CUDA error {}'.format(name, code))


def distribution(values):
    import numpy as np
    return {'mean_ms': float(np.mean(values)), 'p50_ms': float(np.percentile(values, 50)),
            'p95_ms': float(np.percentile(values, 95)), 'p99_ms': float(np.percentile(values, 99)),
            'min_ms': float(min(values)), 'max_ms': float(max(values))}


def resource_summary(samples, baseline):
    def values(name):
        return [s[name] for s in samples if name in s]
    ram, util = values('ram_mb'), values('gpu_util_pct')
    rails = sorted({k for s in samples for k in s.get('power_mw', {})})
    power = {}
    for rail in rails:
        v = [s['power_mw'][rail] / 1000 for s in samples if rail in s.get('power_mw', {})]
        power[rail] = {'mean_w': statistics.mean(v), 'peak_w': max(v), 'samples': len(v)}
    temperatures = {}
    for key in sorted({k for s in samples for k in s.get('temperatures_c', {})}):
        v = [s['temperatures_c'][key] for s in samples if key in s.get('temperatures_c', {})]
        temperatures[key] = {'mean_c': statistics.mean(v), 'peak_c': max(v)}
    soc = [v['peak_c'] for k, v in temperatures.items()
           if k.startswith(('CPU', 'GPU', 'SOC', 'CV')) or k == 'tj']
    total_rail = 'VDD_IN' if 'VDD_IN' in power else 'POM_5V_IN' if 'POM_5V_IN' in power else None
    return {'samples': len(samples), 'baseline_ram_mb': baseline,
            'peak_system_ram_mb': max(ram) if ram else None,
            'peak_ram_above_baseline_mb': max(ram) - baseline if ram and baseline is not None else None,
            'mean_gpu_util_pct': statistics.mean(util) if util else None,
            'power_rails': power, 'total_input_power_rail': total_rail,
            'mean_input_power_w': power[total_rail]['mean_w'] if total_rail else None,
            'peak_soc_temp_c': max(soc) if soc else None, 'temperatures': temperatures}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--duration', type=float, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--allow-unlocked', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import tensorrt as trt
    clocks_before = clock_state()
    if not clocks_locked(clocks_before) and not args.allow_unlocked:
        raise RuntimeError('CPU/GPU clocks are not locked. Run sudo jetson_clocks first.')
    telemetry = Telemetry(args.output)
    telemetry.start()
    buffers, events, stream = [], [], C.c_void_p()
    cuda = None
    try:
        time.sleep(1)
        baseline_values = [s['ram_mb'] for s in telemetry.samples]
        baseline = statistics.median(baseline_values) if baseline_values else None
        cuda = CUDA()
        cuda.call('cudaStreamCreate', C.byref(stream))
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
        if engine is None:
            raise RuntimeError('Failed to deserialize TensorRT engine')
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError('Failed to create TensorRT context')
        bindings, descriptions, outputs = [], [], []
        random = np.random.RandomState(args.seed)
        for i in range(engine.num_bindings):
            shape = tuple(engine.get_binding_shape(i))
            if any(v < 1 for v in shape):
                raise RuntimeError('Static shape required: {}'.format(shape))
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
            size = int(np.prod(shape)) * dtype.itemsize
            ptr = C.c_void_p()
            cuda.call('cudaMalloc', C.byref(ptr), size)
            buffers.append(ptr)
            bindings.append(ptr.value)
            is_input = engine.binding_is_input(i)
            descriptions.append({'name': engine.get_binding_name(i), 'shape': list(shape),
                                 'dtype': str(dtype), 'input': is_input, 'bytes': size})
            if is_input:
                assert shape == (1, 3, 640, 640), shape
                data = random.uniform(0, 1, shape).astype(dtype)
                cuda.call('cudaMemcpy', ptr, C.c_void_p(data.ctypes.data), size, 1)
            else:
                outputs.append((ptr, np.empty(shape, dtype=dtype)))
        inspector = engine.create_engine_inspector()
        (args.output / 'engine_layers.json').write_text(
            inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        for _ in range(2):
            event = C.c_void_p()
            cuda.call('cudaEventCreate', C.byref(event))
            events.append(event)
        start_event, end_event = events

        def enqueue():
            if not context.execute_async_v2(bindings=bindings, stream_handle=stream.value):
                raise RuntimeError('TensorRT execute_async_v2 failed')

        print('Warmup: {} exact iterations'.format(args.warmup), flush=True)
        for _ in range(args.warmup):
            enqueue()
            cuda.call('cudaStreamSynchronize', stream)
        # Validate outputs outside the formal measurement window.
        checks = []
        for ptr, output in outputs:
            cuda.call('cudaMemcpy', C.c_void_p(output.ctypes.data), ptr, output.nbytes, 2)
            finite = bool(np.isfinite(output).all())
            values = output[np.isfinite(output)]
            checks.append({'shape': list(output.shape), 'finite': finite,
                           'nan_count': int(np.isnan(output).sum()),
                           'inf_count': int(np.isinf(output).sum()),
                           'min': float(values.min()) if values.size else None,
                           'max': float(values.max()) if values.size else None})
        (args.output / 'output_validation.json').write_text(json.dumps(checks, indent=2))
        if any(not check['finite'] for check in checks):
            raise RuntimeError('Engine produced non-finite output; see output_validation.json')
        gpu_ms, host_ms, timestamps = [], [], []
        print('Measurement: {} seconds, single stream, serial'.format(args.duration), flush=True)
        begin = time.perf_counter()
        while True:
            iteration_begin = time.perf_counter()
            if iteration_begin - begin >= args.duration:
                break
            cuda.call('cudaEventRecord', start_event, stream)
            enqueue()
            cuda.call('cudaEventRecord', end_event, stream)
            cuda.call('cudaEventSynchronize', end_event)
            iteration_end = time.perf_counter()
            elapsed = C.c_float()
            cuda.call('cudaEventElapsedTime', C.byref(elapsed), start_event, end_event)
            gpu_ms.append(elapsed.value)
            host_ms.append((iteration_end - iteration_begin) * 1000)
            timestamps.append(iteration_end - begin)
        end = time.perf_counter()
        measured = [s for s in telemetry.samples if begin <= s['monotonic_s'] <= end]
        clocks_after = clock_state()
        valid_clocks = clocks_locked(clocks_after) and clocks_before == clocks_after
        result = {'status': 'ok' if valid_clocks or args.allow_unlocked else 'invalid_clocks',
                  'engine': str(args.engine.resolve()), 'tensorrt': trt.__version__,
                  'engine_sha256': sha256(args.engine),
                  'seed': args.seed,
                  'warmup_iterations': args.warmup, 'requested_duration_s': args.duration,
                  'measured_duration_s': end - begin, 'iterations': len(gpu_ms),
                  'measurement_begin_monotonic_s': begin, 'measurement_end_monotonic_s': end,
                  'throughput_fps': len(gpu_ms) / (end - begin),
                  'gpu_compute': distribution(gpu_ms), 'host_latency': distribution(host_ms),
                  'bindings': descriptions, 'output_checks': checks,
                  'clock_state_before': clocks_before, 'clock_state_after': clocks_after,
                  'clocks_locked_and_unchanged': valid_clocks,
                  'resources': resource_summary(measured, baseline),
                  'telemetry_error': telemetry.error,
                  'timing': 'CUDA events; one in-flight enqueue; event completion synchronized each iteration; transfers/preprocess/postprocess excluded'}
        with open(args.output / 'latency.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'elapsed_s', 'gpu_compute_ms', 'host_latency_ms'])
            writer.writerows((i, t, g, h) for i, (t, g, h) in
                            enumerate(zip(timestamps, gpu_ms, host_ms), 1))
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ('status', 'iterations', 'throughput_fps',
                                                'gpu_compute', 'host_latency', 'resources')}, indent=2), flush=True)
        if result['status'] != 'ok':
            raise RuntimeError('Clocks changed during measurement')
    finally:
        telemetry.stop()
        if cuda:
            for event in events:
                cuda.call('cudaEventDestroy', event)
            for ptr in buffers:
                cuda.call('cudaFree', ptr)
            if stream.value:
                cuda.call('cudaStreamDestroy', stream)


if __name__ == '__main__':
    main()
