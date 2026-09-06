# A8 mini RTSP YOLO 检测 V1.2

运行时将 `A8mini_RTSP_YOLO_Detection.py` 和 `rtsp_capture.py` 放在同一目录。使用目标 Orin NX 环境中可运行的 OpenCV、NumPy、Ultralytics、PyTorch/CUDA/TensorRT，以及在相应环境导出的 engine。

```bash
python3 A8mini_RTSP_YOLO_Detection.py --model /path/to/yolo11s.engine --imgsz 640
```

默认地址仍为 `rtsp://192.168.144.25:8554/main.264`。默认先用 GStreamer H.264 硬解码；打开超时 2000 ms，读帧超时 1000 ms，重连间隔 0.2 秒。有线网络可测试 `--read-timeout 800`，代码不会再将它强制放大到 1000 ms。超时过短可能在网络抖动或等待关键帧时引发频繁重连，应以现场日志调整。

```bash
# 无桌面环境，每两秒输出一次统计
python3 A8mini_RTSP_YOLO_Detection.py --model /path/to/yolo11s.engine --no-display

# 单独比较 FFmpeg 后端
python3 A8mini_RTSP_YOLO_Detection.py --model /path/to/yolo11s.engine --backend ffmpeg

# 相机实际输出为 H.265 时，选择相应 GStreamer 解码管线
python3 A8mini_RTSP_YOLO_Detection.py --model /path/to/yolo11s.engine --codec h265

# 显式启用保存完整画面和未经标注的目标裁剪图
python3 A8mini_RTSP_YOLO_Detection.py --model /path/to/yolo11s.engine \
    --save --save-classes person --save-crops --save-cooldown 2
```

`--codec` 只选择本地 GStreamer 解码器，不修改相机配置；FFmpeg 自行探测编码。`--save-classes` 默认 `person`，只有同时指定 `--save` 才保存。保存采用同步磁盘写入，开启后慢磁盘仍会影响检测/显示；采集继续推进，终端统计包含 `SAVE` 耗时。默认关闭保存，不创建图片目录。

## 类别名称显示为 classXX

`class62` 表示已经有类别编号，但名称表缺失或只有占位名称。[Ultralytics 的默认名称逻辑](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py)会在缺少名称时生成 `class0`、`class1` 等标签。

这次检查了目录中的真实文件：`yolo11s.engine` 没有可读取的标准 Ultralytics JSON 元数据头；`yolo11s.pt` 可以读出 80 类名称。已从这份 PT 提取 [yolo11s.names.json](yolo11s.names.json)，请与 engine 放在同一目录。原命令即可自动加载类别表，不必为了名称再额外加载 PT。

名称来源按以下规则选择：

1. 显式指定 `--names-file classes.json` 或 `--names-model original.pt`，二者互斥，优先于内置名称。显式路径错误会报错退出。
2. 未显式指定时，先使用推理结果中有效的内置名称。
3. 内置名称缺失或包含 `classXX` 时，先找 engine 同目录同文件名的 `.names.json`，再找同名 `.pt`。只解析一次；不自动下载权重，也不根据类别数量猜测 COCO。
4. 都没有时警告一次并继续显示编号。类别名称也用于 `--save-classes` 匹配，此时应补齐名称表，或暂用编号。

```bash
# 最轻量：显式指定已从原始 PT 提取的名称表
python3 A8mini_RTSP_YOLO_Detection.py --model yolo11s.engine --names-file yolo11s.names.json

# 延续之前的做法：直接使用原始 PT 中的名称
python3 A8mini_RTSP_YOLO_Detection.py --model yolo11s.engine --names-model yolo11s.pt
```

终端会打印实际名称来源、类别总数和示例。JSON 支持名称数组、从 0 开始的编号字典，或带 `names` 字段的对象。名称表必须来自导出 engine 的原始模型，且编号顺序一致；同样有 80 类并不能证明两个模型的类别定义相同。检测编号超出外部表时会明确报错。

本次 PT 中 `62=tv`、`74=clock`。因此截图中的风扇框会显示 `clock`，这是模型原本的误识别；补名称仅恢复编号对应的标签，不会提高模型识别准确率。

## 采集与重连

```text
A8 mini → 采集子进程 → 单个共享最新帧槽 → 主线程 YOLO / 绘图
             ↑
       独立监控线程：超时、退出、重连、后端切换
```

采集持续读取并覆盖最新帧，推理不会排队追赶应用层积压帧。同一序号只消费一次，超过 `--max-frame-age`（默认 1000 ms）的帧不再送入推理。共享槽默认容量约 24.9 MB，支持最多 3840×2160 个 BGR 像素；可用 `--capture-max-pixels 2073600` 缩小为 1080p 容量。超过容量会明确报错并重连。

子进程使用 `spawn`，不继承主进程 CUDA 状态。采集帧写入共享槽、主线程取得独立快照各有一次必要的内存复制；去掉的是旧版每帧为未启用保存而做的额外 `frame.copy()`。只有本次确实要保存裁剪图时，才另存未经标注的原图。

GStreamer 和 FFmpeg 都在 `VideoCapture` 打开时接收超时参数。GStreamer 同时设置以微秒计的 `tcp-timeout`。不支持带参数构造的 OpenCV 会报告失败，不静默改用无超时构造。

监控线程独立于 YOLO：打开/读帧超过配置时间加 250 ms 调度余量时，终止旧采集进程；若终止无效，进一步 kill。`spawn` 和模块导入另有 10 秒启动预算。旧进程结束后才启动新进程，每次使用新的共享槽和锁，避免继承被终止进程持有的锁。重连恢复还需要进程启动、RTSP 协商、解码首帧时间，不能把 `--read-timeout 1000` 理解为保证一秒内恢复视频。

`auto` 默认连续失败两次切换 GStreamer ↔ FFmpeg（`--backend-failures`）。单次打开失败、达到 `--max-read-failures`、watchdog 超时、进程退出均算一次失败；收到一帧不会清零失败计数，连续接收帧至少 30 秒的连接才视为稳定。稳定后下一次断流从一次失败计起。已稳定的 FFmpeg 不定时回切，避免人为打断视频。显式指定的后端只重连，不切换。

断流时清空旧连接的帧，窗口改为等待新帧的状态，Q/ESC 仍可退出；无桌面使用 Ctrl+C。新连接或长时间无帧后重置 FPS。YOLO 和显示在主线程，模型调用自身若阻塞，窗口仍可能暂时不响应；采集 watchdog 仍可独立运行。

## 如何读统计

| 指标 | 含义 |
| --- | --- |
| CAP | 成功帧对应的一次 `cap.read()` 耗时，包括等待帧；失败读帧耗时单独写入警告 |
| YOLO | `model.predict` 加检测框数据转 CPU 的实际耗时，包括预处理、推理和后处理 |
| AGE | 解码完成到送入 YOLO 的本机等待时间，包含快照复制 |
| FPS | 连续处理结果的平滑吞吐率，不是相机帧率；新连接首帧为 0 |
| SAVE | 本次采样迭代的图片保存耗时；未保存为 0 |

例如 CAP 高而 YOLO 稳定，应先看流输入/解码；YOLO 高则检查推理路径。AGE 不包含相机编码和网络/解码器内部缓冲，不能当作端到端延迟，也不能证明 FFmpeg 内部没有积压。首次模型推理含初始化/预热，不能直接当作稳态性能。

## 文档核对

- [A8 mini 用户手册](<A8 mini云台相机用户手册.pdf>)：印刷页 60–61 说明新旧 RTSP 地址与产品适用范围，A8 mini 使用旧地址；第 19 页说明同一地址最多四路输出。此次没有改相机固件或网络配置。
- [SDK V0.1.1](<SDK - V0.1.1.pdf>)：第 22–23 页，`0x20/0x21` 定义查询/设置编码参数，`VideoEncType` 的 1/2 对应 H264/H265；实际能力仍依赖相机固件。
- [达妙载板说明书](<达妙科技DM-ORIN NX V2.X使用说明书V1.0 (1).pdf>)：介绍载板接口、网口及驱动，不能据此推断已安装 OpenCV 的 GStreamer 支持情况。
- [OpenCV 超时属性](https://docs.opencv.org/4.10.0/d4/d15/group__videoio__flags__base.html)及 [GStreamer 后端源码](https://github.com/opencv/opencv/blob/4.10.0/modules/videoio/src/cap_gstreamer.cpp)：超时需在打开时传入；实际能力取决于 OpenCV/GStreamer 构建版本。
- [GStreamer rtspsrc](https://gstreamer.freedesktop.org/documentation/rtsp/rtspsrc.html#rtspsrc:tcp-timeout)：TCP 超时的单位是微秒；`timeout` 是另一项 UDP 回退属性，不能混用。
- [Ultralytics 固定输入尺寸](https://docs.ultralytics.com/modes/predict/#fixed-shape-vs-minimum-rectangle-rect)：显式使用 `rect=False`。这并不说明旧版固定 engine 必然会收到错误尺寸；仍必须让 `--imgsz` 匹配 engine。

原 review 的核心问题在代码中成立，但“默认允许等 5 秒”只是潜在阻塞路径，不能证明现场每次卡顿都由 RTSP 引起。原来 FPS 的平滑公式也不能单独证明模型耗时增加。V1.2 用分项计时和故障日志帮助区分原因。

## 验证

```bash
python3 -m unittest -v test_rtsp_capture
python3 -m py_compile A8mini_RTSP_YOLO_Detection.py rtsp_capture.py
python3 A8mini_RTSP_YOLO_Detection.py --help
```

测试使用真实 spawn 进程注入打开、读帧和释放阻塞、连续短连接、打开异常；检查自动切换、固定后端重连、退出清理、慢消费者跳帧、共享帧所有权、过期帧、超时参数和保存结果。

本次 12 项测试全部通过，包含主循环的模拟模型推理和保存开关验证；另用本地生成的 MJPEG 视频验证了实际 FFmpeg 带超时参数打开和解码成功，语法检查及 `--help` 通过。

后续类别名称修复增加 8 项测试，现共 20 项通过；覆盖同名 PT/JSON 自动回退、显式覆盖、只加载一次、名称缺失警告、无效/越界类别表，以及本次 PT 提取表中的 62/74 类映射。

当前验证主机 OpenCV 5.0.0 支持 FFmpeg、不支持 GStreamer。本次类别修复实际加载了用户提供的 PT 并提取名称，但未运行 TensorRT engine。未验证 A8 mini 真机网络、Orin NX 硬解码、TensorRT 性能或长时间稳定性。上板后先确认 `cv2.getBuildInformation()` 中 GStreamer 为 YES，且 `gst-inspect-1.0 nvv4l2decoder`、`gst-inspect-1.0 nvvidconv` 可用，再进行实际断网/恢复和长时间运行测试。
