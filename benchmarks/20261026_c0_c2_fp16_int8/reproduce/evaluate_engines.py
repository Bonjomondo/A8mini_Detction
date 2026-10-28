"""Check all four engines on a common, held-out VisDrone validation set."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import tensorrt as trt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, '/home/q/Documents/Detr_Trials/scripts')
from benchmark_trt import CUDA, distribution

root = Path(__file__).resolve().parent
gt_path = root / 'gt_val.json'
gt = COCO(str(gt_path))
images = sorted(gt.dataset['images'], key=lambda x: x['id'])
categories = sorted(c['id'] for c in gt.dataset['categories'])
image_dir = Path('/home/q/Documents/Detr_Trials/dataset/VisDrone2019-DET-YOLO/images/val')
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, '')
for model in ['c0', 'c2']:
    for precision in ['fp16', 'int8']:
        out = root / model / precision
        if (out / 'accuracy.json').exists():
            continue
        cuda = CUDA()
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine((out / 'model.engine').read_bytes())
        context = engine.create_execution_context()
        stream = C.c_void_p()
        cuda.call('cudaStreamCreate', C.byref(stream))
        ptrs, arrays = [], []
        for i in range(engine.num_bindings):
            shape = tuple(engine.get_binding_shape(i))
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i)))
            arr = np.empty(shape, dtype=dtype)
            ptr = C.c_void_p()
            cuda.call('cudaMalloc', C.byref(ptr), arr.nbytes)
            arrays.append(arr)
            ptrs.append(ptr)
        inputs = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
        outputs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
        assert len(inputs) == len(outputs) == 1
        ii, oi = inputs[0], outputs[0]
        predictions, pipeline_ms, scores = [], [], []
        print('EVALUATE', model, precision, flush=True)
        for index, im in enumerate(images):
            path = image_dir / Path(im['file_name']).name
            image = cv2.imread(str(path))
            assert image is not None, path
            assert (image.shape[1], image.shape[0]) == (im['width'], im['height'])
            begin = time.perf_counter()
            resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
            x = np.ascontiguousarray(resized[:, :, ::-1].transpose(2,0,1)[None], dtype=arrays[ii].dtype) / 255.0
            cuda.call('cudaMemcpy', ptrs[ii], C.c_void_p(x.ctypes.data), x.nbytes, 1)
            assert context.execute_async_v2(bindings=[p.value for p in ptrs], stream_handle=stream.value)
            cuda.call('cudaStreamSynchronize', stream)
            y = arrays[oi]
            cuda.call('cudaMemcpy', C.c_void_p(y.ctypes.data), ptrs[oi], y.nbytes, 2)
            assert np.isfinite(y).all(), (model, precision, path)
            values = y[0].astype(np.float32)
            probs = values[:, 4:]
            labels = probs.argmax(axis=1)
            confidence = probs.max(axis=1)
            selected = np.argsort(-confidence)[:300]
            w, h = im['width'], im['height']
            for j in selected:
                if confidence[j] < 0.001:
                    continue
                cx, cy, bw, bh = values[j, :4]
                x1, y1 = max(0.,float((cx-bw/2)*w)), max(0.,float((cy-bh/2)*h))
                x2, y2 = min(float(w),float((cx+bw/2)*w)), min(float(h),float((cy+bh/2)*h))
                predictions.append({'image_id':im['id'], 'category_id':categories[int(labels[j])],
                                    'bbox':[x1,y1,max(0.,x2-x1),max(0.,y2-y1)], 'score':float(confidence[j])})
            pipeline_ms.append((time.perf_counter()-begin)*1000)
            scores.append(float(confidence.max()))
            if index % 100 == 0:
                print(model, precision, index+1, '/', len(images), flush=True)
        pred_path = out / 'pred_val.json'
        pred_path.write_text(json.dumps(predictions, separators=(',',':')))
        dt = gt.loadRes(predictions)
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.maxDets = [1,10,100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        names = ['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl']
        report = {'model':model, 'precision':precision, 'images':len(images),
                  'all_outputs_finite':True, 'detections':len(predictions),
                  'metrics_percent':dict(zip(names,[float(v*100) for v in evaluator.stats])),
                  'gt_sha256':hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                  'pipeline_excluding_decode':distribution(pipeline_ms[10:]),
                  'mean_max_class_score':float(np.mean(scores)),
                  'protocol':'548 held-out images; direct RGB stretch640; /255; one argmax class/query; conf>=0.001; no NMS; clipped boxes; COCOeval maxDets100. Preprocessing may differ from training validator; compare formats within this evaluation only.'}
        (out / 'accuracy.json').write_text(json.dumps(report, indent=2))
        print('AP_RESULT',model,precision,report['metrics_percent']['AP'],flush=True)
        for ptr in ptrs:
            cuda.call('cudaFree',ptr)
        cuda.call('cudaStreamDestroy',stream)
        del context, engine, runtime
