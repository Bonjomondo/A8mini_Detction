# Orin NX Test A / Test C 实测结果

统一 640×640、Batch 1、静态 TensorRT FP16；每次预热 500 次，正式 30 秒，重复 3 次。电源模式、锁频和实际运行参数见 [benchmark_env.md](benchmark_env.md)。

Test A 用 CUDA events 测 GPU compute，主机计时测 enqueue 到同步完成；单 stream、每次仅一个推理在执行。FPS 是完整串行计时循环的实测吞吐，包含必要的主机提交、事件同步和计时开销，不能用 1000/GPU time 替代。

| Model | Params (M) | Estimated GFLOPs | GPU latency (ms) | Serial FPS | Peak system RAM (MiB) | Mean VDD_IN (W) | Peak SoC (°C) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 2.62 | 6.48 | 9.403 | 105.97 | 7720 | 11.06 | 64.12 |
| YOLO11s | 9.44 | 21.47 | 15.583 | 63.99 | 7039 | 11.94 | 65.12 |
| YOLO11m | 20.09 | 67.99 | 31.100 | 32.10 | 7605 | 12.75 | 66.25 |
| YOLO26n | 2.41 | 5.36 | 9.662 | 103.10 | 7615 | 10.94 | 63.22 |
| YOLO26s | 9.50 | 20.69 | 16.291 | 61.22 | 8062 | 11.76 | 63.91 |
| YOLO26m | 20.41 | 68.19 | 31.470 | 31.73 | 6645 | 12.61 | 64.91 |
| RT-DETR-R18 | 19.98 | 59.72 | 33.199 | 30.09 | 7015 | 11.38 | 64.34 |
| SO-DETR-NoFFT | 20.36 | 63.71 | 38.720 | 25.80 | 7083 | 11.36 | 64.59 |

| Model | GPU mean ± run SD (ms) | P50 | P95 | P99 | Host latency (ms) | Serial FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 9.403 ± 0.023 | 9.403 | 9.433 | 9.444 | 9.422 | 105.97 |
| YOLO11s | 15.583 ± 0.011 | 15.572 | 15.623 | 16.026 | 15.607 | 63.99 |
| YOLO11m | 31.100 ± 0.029 | 31.095 | 31.159 | 31.491 | 31.128 | 32.10 |
| YOLO26n | 9.662 ± 0.027 | 9.655 | 9.699 | 9.717 | 9.683 | 103.10 |
| YOLO26s | 16.291 ± 0.026 | 16.283 | 16.336 | 16.369 | 16.315 | 61.22 |
| YOLO26m | 31.470 ± 0.051 | 31.460 | 31.556 | 31.715 | 31.498 | 31.73 |
| RT-DETR-R18 | 33.199 ± 0.004 | 33.179 | 33.290 | 33.742 | 33.218 | 30.09 |
| SO-DETR-NoFFT | 38.720 ± 0.011 | 38.686 | 38.908 | 39.340 | 38.741 | 25.80 |

Test C 与正式推理同时采样。RAM 为系统总占用，不等于模型独占显存；ΔRAM 为相对每次加载 engine 前的系统 RAM 的最大增加量。功耗为 VDD_IN 模块输入功率，不是 25W 模式名称，也不是载板电源插口的整机功率。

| Model | Peak system RAM (MiB) | ΔRAM (MiB) | GPU util (%) | Mean VDD_IN (W) | Peak SoC temp (°C) |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 7720 | 2025 | 97.9 | 11.06 | 64.12 |
| YOLO11s | 7039 | 1273 | 97.8 | 11.94 | 65.12 |
| YOLO11m | 7605 | 1240 | 98.4 | 12.75 | 66.25 |
| YOLO26n | 7615 | 1241 | 97.5 | 10.94 | 63.22 |
| YOLO26s | 8062 | 1651 | 98.0 | 11.76 | 63.91 |
| YOLO26m | 6645 | 1239 | 98.7 | 12.61 | 64.91 |
| RT-DETR-R18 | 7015 | 845 | 97.7 | 11.38 | 64.34 |
| SO-DETR-NoFFT | 7083 | 861 | 97.7 | 11.36 | 64.59 |

| Model | Deploy params (M) | Estimated GFLOPs | ONNX (MiB) | Engine (MiB) |
|---|---:|---:|---:|---:|
| YOLO11n | 2.62 | 6.48 | 10.21 | 7.42 |
| YOLO11s | 9.44 | 21.47 | 36.25 | 20.47 |
| YOLO11m | 20.09 | 67.99 | 76.89 | 41.11 |
| YOLO26n | 2.41 | 5.36 | 9.43 | 7.01 |
| YOLO26s | 9.50 | 20.69 | 36.47 | 20.76 |
| YOLO26m | 20.41 | 68.19 | 78.11 | 41.82 |
| RT-DETR-R18 | 19.98 | 59.72 | 76.75 | 41.51 |
| SO-DETR-NoFFT | 20.36 | 63.71 | 77.93 | 42.25 |

统计口径：延迟分位数由 3 次全部逐次样本合并计算；SD 是 3 个 run mean 的样本标准差；功耗/利用率由全部正式 tegrastats 样本合并。THOP 估算 GFLOPs = MACs×2，不支持的算子未计入；部署参数量在融合/移除训练辅助分支后统计。

YOLO26 保留训练好的 one-to-one 分支，仅排除最终 top-k 检测筛选；RT-DETR 官方权重 strict=True 加载，导出原始 boxes/scores；NoFFT 使用项目 SO_DETR fork。模型内置解码及 RT-DETR 的 query selection 属于网络图，保留计时。FP16 是 builder 开启的精度，TensorRT 允许必要的 FP32 回退；FP32 I/O，TF32 关闭。

类别数：YOLO 和官方 RT-DETR 为 COCO 80 类，NoFFT 为 VisDrone 10 类。这是现有权重的部署性能基线，不用于隔离类别数影响或比较检测精度。

RT-DETR 来源：[官方实现](https://github.com/lyuwenyu/RT-DETR)，提交号保存在该模型 model_info.json。

完整运行目录留在远程机，含导出、构建日志和每轮 tegrastats.log；Git 跟踪配置、模型信息、每轮逐次 latency.csv、带单调时间戳的 telemetry.jsonl 和结果 JSON。未执行 Test B、AP、相机或 INT8。

本次恢复记录：初次导出的五个 YOLO ONNX 含有 NaN 卷积权重，保留在远程主机的 `archive/` 目录，未纳入 Git。重新导出使用 PyTorch 的逐通道 Conv/BN 融合；构建前检查 ONNX 权重有限值，每次预热后检查全部 engine 输出。YOLO11n 复用原有完整三轮结果，其余模型使用恢复后的环境快照；各模型 config.json 与各 run/command.json 可追溯实际快照。

跨日环境记录：2026-10-06（YOLO11n、YOLO11s、YOLO11m、YOLO26n、YOLO26s）；2026-10-07（YOLO26m、RT-DETR-R18、SO-DETR-NoFFT）。CPU/GPU 锁频条件一致；全系统 RAM 的绝对峰值还受桌面与后台进程影响，应结合各 run 的加载前 baseline_ram_mb 和 ΔRAM 判断资源占用。
