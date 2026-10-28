import hashlib
import json
from pathlib import Path

out = Path('outputs/orin_c0_c2_fp16_int8')
data = json.loads((out/'summary.json').read_text())
rows = data['rows']
by = {(r['model'],r['precision']):r for r in rows}
precision_evidence = {}
for model in ['c0','c2']:
    for precision in ['fp16','int8']:
        key=model+'/'+precision
        layers=json.loads((out/key/'repeat1/engine_layers.json').read_text())['Layers']
        conv=[l for l in layers if 'CONV' in l.get('LayerType','').upper()]
        def uses_int8(layer):
            tensors=layer.get('Inputs',[])+layer.get('Outputs',[])
            return any('INT8' in t.get('Format/Datatype','').upper() for t in tensors)
        precision_evidence[key]={'layers_total':len(layers), 'layers_with_int8_tensor_io':sum(map(uses_int8,layers)),
                                 'convolution_layers':len(conv),'convolutions_with_int8_tensor_io':sum(map(uses_int8,conv))}
        if precision=='int8':
            assert precision_evidence[key]['convolutions_with_int8_tensor_io']>0
            assert data['evidence'][key]['build']['calibration_batches_consumed']==256
(out/'precision_evidence.json').write_text(json.dumps(precision_evidence,indent=2))
lines=['# Orin NX：C0/C2 的 FP16 与 INT8 实测对比','',
       '测试日期：2026-10-26（Asia/Shanghai）。四组均使用同一块 Orin NX、相同输入与计时程序。','',
       '## 延迟结果','',
       '| 模型 | 精度配置 | GPU均值 ms | P50 ms | P95 ms | 串行实测 FPS | 三轮均值范围 ms |',
       '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['model']} | {r['precision']} | {r['gpu_mean_ms']:.3f} | {r['gpu_p50_ms']:.3f} | {r['gpu_p95_ms']:.3f} | {r['serial_throughput_fps']:.2f} | {r['repeat_mean_min_ms']:.3f}–{r['repeat_mean_max_ms']:.3f} |")
lines += ['', 'GPU延迟由CUDA事件计时；单stream、每次仅一个在途推理并等待完成。计时不含读图、CPU预处理、H2D/D2H传输及检测后处理，也不含相机、网络或视频解码。FPS是整个串行计时循环的实测吞吐，含事件与主机循环开销。', '']
for model in ['C0','C2']:
    fp,it=by[(model,'FP16')],by[(model,'INT8')]
    lines.append(f"- {model}：INT8相对FP16加速 **{fp['gpu_mean_ms']/it['gpu_mean_ms']:.3f}×**，平均GPU延迟减少 **{(1-it['gpu_mean_ms']/fp['gpu_mean_ms'])*100:.2f}%**；同协议COCO AP变化 **{it['AP_percent']-fp['AP_percent']:+.3f}个百分点**。")
for precision in ['FP16','INT8']:
    c0,c2=by[('C0',precision)],by[('C2',precision)]
    lines.append(f"- {precision}下，C2相对C0的平均GPU延迟变化 **{(c2['gpu_mean_ms']/c0['gpu_mean_ms']-1)*100:+.2f}%**。")
lines.append(f"- 本次协议下，C2 INT8相对C0 FP16加速 **{by[('C0','FP16')]['gpu_mean_ms']/by[('C2','INT8')]['gpu_mean_ms']:.3f}×**，COCO AP高 **{by[('C2','INT8')]['AP_percent']-by[('C0','FP16')]['AP_percent']:.3f}个百分点**。")
lines += ['', '## 同一验证集的精度检查','',
          '| 模型 | 精度配置 | COCO AP % | AP50 % | APs % | engine MiB |',
          '|---|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['model']} | {r['precision']} | {r['AP_percent']:.3f} | {r['AP50_percent']:.3f} | {r['APs_percent']:.3f} | {r['engine_MiB']:.2f} |")
lines += ['', '548张VisDrone验证图像逐文件SHA256与4090上的图像一致；使用从4090复制的正式COCO标注。直接拉伸为640×640、BGR转RGB、/255；每个query取最高分类分数，conf≥0.001，无NMS，保留至多300个query，框裁剪到图像内；COCOeval maxDets=[1,10,100]。四组均检查输出有限。', '',
          '本次直接缩放的实现可能与训练框架验证器的图像加载/二次缩放路径不同，因此本表用于本次四种引擎之间比较；不把它直接当作重现训练报告AP的证据。训练服务器原有精度记录另存于[training_reference](training_reference/)。', '',
          '## 硬件、软件与协议','',
          '- 目标：`ssh q`，`q@192.168.2.110`，NVIDIA Orin NX Developer Kit，aarch64，16 GB共享内存。',
          '- Jetson Linux R35.6.4；TensorRT 8.5.2.2；导出用系统PyTorch 2.1.0a0+41361538.nv23.06。完整[环境记录](environment.txt)。',
          '- 保留原25W模式（id=3）；测试时GPU408 MHz、CPU1497.6 MHz，EMC锁频。结果仅代表此模式与频率。每轮验证CPU/GPU锁频且前后不变；测试结束恢复原频率设置。',
          '- 静态输入`1×3×640×640`，输出`1×300×14`，10类；每组精确预热500次，正式30秒，重复3轮；第二轮反向排列四组，减轻顺序/温度影响。',
          '- 共12个正式测量窗口；逐次CUDA事件计时CSV和结果JSON随报告保留。均值/分位数由全部正式样本合并计算；三轮均值范围单独列出。',
          '- 正式延迟测试采用seed=0、[0,1]均匀分布的固定随机输入，预先驻留GPU；真实548张图像用于独立的精度评估。',
          '- [正式窗口遥测核验](telemetry_audit.json)确认GPU实际频率始终408 MHz，最高SoC温度66.937°C；[恢复核验](restore_verification.json)确认原有可读频率设置均已恢复。',
          '- TensorRT工作空间上限2 GiB；FP16启用；INT8额外启用INT8并允许FP16/FP32回退。构建日志提示部分候选实现超出工作空间，这限定了本次算子搜索范围。',
          '- INT8是EntropyCalibrator2后训练量化：从VisDrone训练集确定性抽取256张，seed=0、batch=1，两模型各自校准；不用验证图像或随机尺度代替校准。见[校准图像清单与SHA256](calibration_images.json)。',
          '- 本次引擎属于TensorRT混合精度实现，不能把“INT8”解释为所有算子都以INT8执行；逐层实际格式见[精度证据](precision_evidence.json)及各组`engine_layers.json`。[NVIDIA量化文档](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html)。', '',
          '## 权重与源码身份','',
          '来源：4090服务器`/home/zjnu/文档/ZHJ_WorkSpace/ultralytics`，分支`feat/r18-p2-sfi`，commit `eb40ae103fcd4ff9dd03a8d356022c2fbe5ba5f1`。存在的报告改动未修改；本次把实际推理包冻结到Orin的独立source目录。', '',
          '权重目录：`runs/detect/r18_p2_sfi_convergence_20261024/rtdetr-r18-{c0,c2}-seed0-cont/weights/best.pt`。两组累计完成150＋350轮，但训练报告的经验平台判据均为False；此处称为“最新最终best”，不宣称已满足收敛判据。', '',
          '| 模型 | best.pt SHA256 |', '|---|---|',
          '| C0 | `021596fbecd5d2edc5e1b5d5477f84087d4caf48bc2efe274e08bc047dc5e393` |',
          '| C2 | `dd6a761f42617ac34784f657e938bd9082be363df6be1fc505e77f11f0f8fa32` |', '',
          '导出中发现项目CPU融合路径产生非有限输出；保留原始权重，改用未融合评估模型导出，由ONNX常量折叠/TensorRT处理融合。导出输入/输出和ONNX浮点初始化张量均通过有限值检查；正式引擎输出另行验证。', '',
          '## 文件与复现','',
          '- [汇总CSV](summary.csv)；[完整汇总JSON](summary.json)；[输入来源清单](input_manifest.json)；[验证图像身份](validation_image_manifest.json)。',
          '- 各模型目录包含ONNX/权重身份、引擎构建信息、三个重复实验的原始计时、COCO评估指标和逐层精度信息。',
          '- 完整模型、ONNX、engine、校准缓存、逐图预测及构建日志保留在Orin：`/home/q/Documents/Detr_Trials/benchmarks/20261026_c0_c2_fp16_int8/`。engine位于`{c0,c2}/{fp16,int8}/model.engine`。',
          '- 复现脚本见[reproduce](reproduce/)；原始训练仓库未改动。仅重新汇总无需GPU；重新构建/推理需要所列外部权重、源码、数据集与对应Jetson环境。',
          '- 本次测的是GPU推理与离线精度；未接入实时相机或RTSP链路。', '']
(out/'report.md').write_text('\n'.join(lines))
print(out/'report.md')
