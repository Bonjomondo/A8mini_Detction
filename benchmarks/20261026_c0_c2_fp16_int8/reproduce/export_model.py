import argparse
import hashlib
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--model', required=True)
a = p.parse_args()
root = Path(a.root)
sys.path.insert(0, str(root / 'source'))
import numpy as np
import torch
import onnx
torch.set_num_threads(4)
torch.manual_seed(0)
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as tu
def fold(conv, bn):
    return torch.nn.utils.fuse_conv_bn_eval(conv, bn).requires_grad_(False)
tasks.fuse_conv_and_bn = fold
tu.fuse_conv_and_bn = fold
folder = root / a.model
path = folder / 'best.pt'
ckpt = torch.load(str(path), map_location='cpu')
net = (ckpt.get('ema') or ckpt['model']).float().eval()
params = sum(x.numel() for x in net.parameters())
# Export the unfused evaluation model; ONNX/TRT performs Conv/BN folding.
# This avoids a non-finite result in this fork's CPU reparameterization path.
head = net.model[-1]
head.export, head.dynamic, head.format = True, False, 'onnx'
x = torch.rand(1, 3, 640, 640)
with torch.no_grad():
    y = net(x)
    assert torch.isfinite(y).all()
    torch.onnx.export(net, x, str(folder / 'model.onnx'), opset_version=16,
                      input_names=['images'], output_names=['output0'],
                      dynamic_axes=None, do_constant_folding=True)
graph = onnx.load(str(folder / 'model.onnx'))
onnx.checker.check_model(graph)
for tensor in graph.graph.initializer:
    val = onnx.numpy_helper.to_array(tensor)
    if val.dtype.kind == 'f':
        assert np.isfinite(val).all(), tensor.name
info = {'model': a.model, 'torch': torch.__version__, 'parameters': params,
        'weights_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'onnx_sha256': hashlib.sha256((folder / 'model.onnx').read_bytes()).hexdigest(),
        'input_shape': list(x.shape), 'output_shape': list(y.shape),
        'head': str(type(head)), 'checkpoint_epoch': ckpt.get('epoch'),
        'preprocessing': 'RGB resize directly to 640x640, NCHW float32 /255; RTDETR stretch',
        'opset': 16, 'output_finite': True,
        'fusion': 'unfused PyTorch evaluation graph; ONNX constant folding and TensorRT optimization'}
(folder / 'model_info.json').write_text(json.dumps(info, indent=2))
print(json.dumps(info, indent=2), flush=True)
