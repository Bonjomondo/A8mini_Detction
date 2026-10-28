import argparse
import ctypes as C
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, '/home/q/Documents/Detr_Trials/scripts')
from benchmark_trt import CUDA

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--model', required=True)
p.add_argument('--precision', choices=['fp16', 'int8'], required=True)
a = p.parse_args()
root = Path(a.root)
folder = root / a.model
out = folder / a.precision
out.mkdir(exist_ok=True)
logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, '')

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self):
        super().__init__()
        self.cuda = CUDA()
        self.ptr = C.c_void_p()
        self.cuda.call('cudaMalloc', C.byref(self.ptr), 1*3*640*640*4)
        self.images = json.loads((root / 'calibration_images.json').read_text())
        self.index = 0
    def get_batch_size(self):
        return 1
    def get_batch(self, names):
        if self.index == len(self.images):
            return None
        image = cv2.imread(self.images[self.index]['path'])
        if image is None:
            raise RuntimeError('Unreadable calibration image')
        image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        self.cuda.call('cudaMemcpy', self.ptr, C.c_void_p(x.ctypes.data), x.nbytes, 1)
        self.index += 1
        if self.index % 32 == 0:
            print('CALIBRATION', self.index, len(self.images), flush=True)
        return [self.ptr.value]
    def read_calibration_cache(self):
        cache = out / 'calibration.cache'
        return cache.read_bytes() if cache.exists() else None
    def write_calibration_cache(self, cache):
        (out / 'calibration.cache').write_bytes(cache)

builder = trt.Builder(logger)
builder.max_threads = 4
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
assert parser.parse((folder / 'model.onnx').read_bytes()), '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
config.set_flag(trt.BuilderFlag.FP16)
calibrator = None
if a.precision == 'int8':
    assert builder.platform_has_fast_int8
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = Calibrator()
    config.int8_calibrator = calibrator
cache_path = folder / 'timing.cache'
seed_cache = cache_path if cache_path.exists() else root / 'c0' / 'timing.cache'
cache = config.create_timing_cache(seed_cache.read_bytes() if seed_cache.exists() else b'')
config.set_timing_cache(cache, ignore_mismatch=False)
start = time.time()
serialized = builder.build_serialized_network(network, config)
assert serialized is not None, 'TensorRT build failed'
target = out / 'model.engine'
target.write_bytes(serialized)
cache_path.write_bytes(config.get_timing_cache().serialize())
info = {'precision_requested': a.precision, 'fp16_fallback': True,
        'calibration': 'EntropyCalibrator2; 256 deterministic VisDrone train images; RGB stretch /255' if calibrator else None,
        'calibration_batches_consumed': calibrator.index if calibrator else 0,
        'tensorrt': trt.__version__, 'build_seconds': time.time() - start,
        'engine_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'engine_bytes': target.stat().st_size, 'workspace_bytes': 2 << 30}
(out / 'build_info.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2), flush=True)
if calibrator:
    calibrator.cuda.call('cudaFree', calibrator.ptr)
