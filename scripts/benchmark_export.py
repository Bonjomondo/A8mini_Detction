#!/usr/bin/env python3
"""Export the supplied detection weights for the Orin Test A/C benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = '29320b6fd828f8e0987a71426cf2d961b09dfed7'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['yolo', 'nofft', 'rtdetr'], required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.weights = args.weights.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/orin-benchmark-matplotlib')
    import torch
    import onnx
    import numpy as np
    torch.set_num_threads(4)
    torch.manual_seed(0)
    info = {'weights': str(args.weights), 'weights_sha256': sha256(args.weights),
            'input_shape': [1, 3, 640, 640], 'opset': 16,
            'postprocessing': 'External detection selection/NMS excluded; box decode retained'}
    x = torch.rand(1, 3, 640, 640)
    target = args.output / 'model.onnx'
    if args.kind == 'rtdetr':
        official = ROOT / 'benchmarks/vendor/RT-DETR/rtdetr_pytorch'
        if not official.exists():
            subprocess.run(['git', 'clone', 'https://github.com/lyuwenyu/RT-DETR.git',
                            str(official.parent)], check=True)
            subprocess.run(['git', '-C', str(official.parent), 'checkout', OFFICIAL_COMMIT], check=True)
        sys.path.insert(0, str(official))
        # Import only official inference modules. The upstream package __init__
        # eagerly imports training data transforms for an older torchvision API.
        # Namespace packages avoid those unrelated imports without changing models.
        import types
        for name in ('src', 'src.nn', 'src.nn.backbone', 'src.zoo', 'src.zoo.rtdetr'):
            package = types.ModuleType(name)
            package.__path__ = [str(official / name.replace('.', '/'))]
            sys.modules[name] = package
        from src.nn.backbone.presnet import PResNet
        from src.zoo.rtdetr.hybrid_encoder import HybridEncoder, RepVggBlock
        from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
        from src.zoo.rtdetr.rtdetr import RTDETR
        from src.core import YAMLConfig
        cfg = YAMLConfig(str(official / 'configs/rtdetr/rtdetr_r18vd_6x_coco.yml'))
        cfg.yaml_cfg['PResNet']['pretrained'] = False
        cfg.yaml_cfg['num_classes'] = 80
        network = cfg.model
        checkpoint = torch.load(str(args.weights), map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        network.load_state_dict(state, strict=True)
        info['params'] = sum(v.numel() for v in network.parameters())
        network.deploy()
        # Upstream deploy keeps the now-inactive RepVGG training convolutions.
        # Drop them after their kernels have been fused; the eval forward uses conv.
        for module in network.modules():
            if isinstance(module, RepVggBlock) and hasattr(module, 'conv'):
                del module.conv1
                del module.conv2
        info['parameter_note'] = 'Checkpoint total and fused module total saved separately; denoising embedding retained but unused during eval'

        class RawDetector(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images):
                output = self.model(images)
                return torch.cat((output['pred_boxes'], output['pred_logits'].sigmoid()), -1)

        model = RawDetector(network).eval()
        info['source'] = 'https://github.com/lyuwenyu/RT-DETR'
        info['source_commit'] = subprocess.check_output(
            ['git', '-C', str(official.parent), 'rev-parse', 'HEAD'], text=True).strip()
        info['implementation'] = 'official RT-DETR PyTorch; strict checkpoint load; raw boxes/scores'
        torch.onnx.export(model, x, str(target), input_names=['images'],
                          output_names=['output0'], opset_version=16, dynamic_axes=None)
    else:
        if args.kind == 'nofft':
            sys.path.insert(0, str(ROOT / 'SO_DETR'))
            from ultralytics import RTDETR as Loader
            info['source_files_sha256'] = {name: sha256(ROOT / 'SO_DETR/ultralytics' / name)
                for name in ('nn/tasks.py', 'nn/uav_modules/block.py',
                             'nn/modules/block.py', 'nn/modules/head.py', 'engine/exporter.py')}
        else:
            from ultralytics import YOLO as Loader
            # Keep the trained YOLO26 one-to-one inference head. Only bypass
            # final top-k/gather selection, so Test A excludes detector postprocess.
            # Setting end2end=False would incorrectly switch to the one-to-many head.
            from ultralytics.nn.modules import Detect
            def raw_postprocess(self, preds):
                return preds
            Detect.postprocess = raw_postprocess
        import ultralytics
        # Use PyTorch's channel-wise BN folding instead of a dense diagonal
        # matrix multiplication. The latter produced NaN weights on this host
        # during CPU export, despite a finite original checkpoint.
        import ultralytics.nn.tasks as tasks
        import ultralytics.utils.torch_utils as torch_utils
        def fuse_conv_bn(conv, bn):
            return torch.nn.utils.fuse_conv_bn_eval(conv, bn).requires_grad_(False)
        tasks.fuse_conv_and_bn = fuse_conv_bn
        torch_utils.fuse_conv_and_bn = fuse_conv_bn
        info['conv_bn_fusion'] = 'torch.nn.utils.fuse_conv_bn_eval; channel-wise scaling'
        info['ultralytics'] = ultralytics.__version__
        info['ultralytics_path'] = ultralytics.__file__
        local = args.output / args.weights.name
        if not local.exists():
            shutil.copy2(args.weights, local)
        loaded = Loader(str(local))
        bad_state = [name for name, value in loaded.model.state_dict().items()
                     if value.is_floating_point() and not torch.isfinite(value).all()]
        if bad_state:
            raise RuntimeError('Non-finite original checkpoint tensors: ' + ', '.join(bad_state))
        info['params'] = sum(v.numel() for v in loaded.model.parameters())
        exported = loaded.export(format='onnx', imgsz=640, batch=1, dynamic=False,
                                 half=False, opset=16, simplify=False, device='cpu')
        exported = Path(exported)
        if exported.resolve() != target.resolve():
            shutil.move(str(exported), str(target))
        local.unlink()
        model = loaded.model.float().eval()
        try:
            model.fuse()
        except TypeError:
            model.fuse(verbose=False)
        info['implementation'] = ('SO-DETR fork, raw query boxes/scores' if args.kind == 'nofft'
                                  else 'Ultralytics; YOLO26 raw one-to-one head; no final top-k/NMS')
    info['deploy_params'] = sum(v.numel() for v in model.parameters())
    try:
        from thop import profile
        with torch.no_grad():
            macs, _ = profile(model, inputs=(x,), verbose=False)
        info['gflops'] = 2 * macs / 1e9
        info['gflops_method'] = 'THOP MACs x 2, deployed FP32 PyTorch model at 640; unsupported ops not counted'
    except Exception as e:
        info['gflops'] = None
        info['gflops_error'] = repr(e)
    graph = onnx.load(str(target))
    onnx.checker.check_model(graph)
    bad_initializers = []
    for tensor in graph.graph.initializer:
        values = onnx.numpy_helper.to_array(tensor)
        if values.dtype.kind == 'f' and not np.isfinite(values).all():
            bad_initializers.append(tensor.name)
    if bad_initializers:
        raise RuntimeError('Non-finite ONNX weights: ' + ', '.join(bad_initializers))
    info['onnx_initializers_finite'] = True
    info['onnx_inputs'] = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
                           for v in graph.graph.input}
    info['onnx_outputs'] = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
                            for v in graph.graph.output}
    assert info['onnx_inputs'] == {'images': [1, 3, 640, 640]}, info['onnx_inputs']
    info['onnx_sha256'] = sha256(target)
    info['onnx_bytes'] = target.stat().st_size
    (args.output / 'model_info.json').write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
