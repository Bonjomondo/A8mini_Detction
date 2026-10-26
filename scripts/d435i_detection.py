#!/usr/bin/env python3
"""D435i RGB 最新帧检测及稳态性能测试（使用项目内 SO_DETR）。"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "weights" / "noFFN_fp16.engine"))
    parser.add_argument("--model-type", choices=("rtdetr", "yolo"), default="rtdetr")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--serial", default="", help="多台 RealSense 时指定序列号")
    parser.add_argument("--duration", type=float, default=60, help="预热后测试秒数；0 持续运行")
    parser.add_argument("--warmup", type=int, default=10, help="不计入统计的预热帧数")
    parser.add_argument("--no-display", action="store_true", help="关闭绘图和窗口，测试处理吞吐")
    parser.add_argument("--capture-only", action="store_true", help="只测试 RGB 采集，不加载模型")
    parser.add_argument("--output", default="", help="结果目录，默认 test_results/d435i/时间戳")
    args = parser.parse_args()
    for field in ("width", "height", "fps", "imgsz"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field} 必须大于 0")
    if args.warmup < 0 or not 0 <= args.conf <= 1:
        parser.error("warmup 必须非负，conf 必须在 0 到 1 之间")
    if not 0 <= args.duration < float("inf"):
        parser.error("duration 必须是非负有限数值")
    if not args.no_display and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        parser.error("没有桌面显示环境，请加 --no-display")
    return args


class LatestColorCapture:
    """采集线程持续排空 SDK；消费后独占 ndarray，不累积待处理帧。"""

    def __init__(self, rs, np, args):
        self.rs, self.np = rs, np
        self.pipeline = rs.pipeline()
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.packet = None
        self.error = None
        self.received = 0
        self.sdk_gaps = 0
        self.thread = None
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        profile = self.pipeline.start(config)
        device = profile.get_device()
        video = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.info = {
            "name": device.get_info(rs.camera_info.name),
            "usb": device.get_info(rs.camera_info.usb_type_descriptor),
            "width": video.width(), "height": video.height(), "fps": video.fps(),
            "streams": "RGB BGR8 only; depth and IMU disabled",
        }
        self.thread = threading.Thread(target=self._run, name="realsense-color", daemon=True)
        self.thread.start()

    def _run(self):
        previous_number = None
        try:
            while not self.stop.is_set():
                frames = self.pipeline.wait_for_frames(1000)
                captured_at = time.perf_counter()
                color = frames.get_color_frame()
                if not color:
                    continue
                number = color.get_frame_number()
                frame = self.np.asanyarray(color.get_data()).copy()
                with self.condition:
                    self.received += 1
                    if previous_number is not None:
                        self.sdk_gaps += max(0, number - previous_number - 1)
                    previous_number = number
                    self.packet = (self.received, frame, captured_at)
                    self.condition.notify_all()
        except Exception as error:
            with self.condition:
                self.error = error
                self.condition.notify_all()

    def take(self, previous_sequence, timeout=2):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.error is not None or
                (self.packet is not None and self.packet[0] != previous_sequence), timeout)
            if self.error is not None:
                raise RuntimeError(f"D435i 采集失败: {self.error}") from self.error
            if not ready:
                raise TimeoutError("D435i 两秒内没有新 RGB 帧")
            return self.packet

    def counters(self):
        with self.condition:
            return self.received, self.sdk_gaps

    def close(self):
        self.stop.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=2)
        finally:
            self.pipeline.stop()


def describe(np, samples):
    return {"mean": float(np.mean(samples)), "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)), "max": float(np.max(samples))}


def benchmark(args):
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    model = torch = None
    predict_args = dict(imgsz=args.imgsz, conf=args.conf, device=args.device,
                        verbose=False, save=False, show=False)
    if not args.capture_only:
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        sys.path.insert(0, str(ROOT / "SO_DETR"))
        # 防止库导入时尝试自动安装依赖或访问在线资源。
        os.environ["YOLO_AUTOINSTALL"] = "false"
        import torch
        from ultralytics import RTDETR, YOLO

        if args.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请在可访问 Jetson GPU 的环境中执行")
        model = (RTDETR if args.model_type == "rtdetr" else YOLO)(str(model_path))

    capture = None
    output = Path(args.output).expanduser() if args.output else (
        ROOT / "test_results" / "d435i" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    started = None
    samples = []
    previous_sequence = 0
    skipped = 0
    last_result = last_frame = None
    interrupted = False
    try:
        capture = LatestColorCapture(rs, np, args)
        print(f"[INFO] {capture.info}", flush=True)
        print(f"[INFO] 模型: {args.model}; 预热 {args.warmup} 帧; 测试 {args.duration}s", flush=True)
        initial_received, initial_gaps = capture.counters()
        for _ in range(args.warmup):
            previous_sequence, frame, _ = capture.take(previous_sequence)
            if model is not None:
                result = model.predict(frame, **predict_args)[0]
                result.boxes.data.cpu().numpy()
        if torch is not None and args.device != "cpu":
            torch.cuda.synchronize(model.predictor.device)
        initial_received, initial_gaps = capture.counters()
        # 将预热期间的帧跳过；测试从下一张新帧开始。
        previous_sequence = initial_received
        started = time.perf_counter()
        last_log = started
        while not args.duration or time.perf_counter() - started < args.duration:
            tick = time.perf_counter()
            sequence, frame, captured_at = capture.take(previous_sequence)
            skipped += max(0, sequence - previous_sequence - 1)
            previous_sequence = sequence
            infer_start = time.perf_counter()
            age_ms = (infer_start - captured_at) * 1000
            predict_ms = 0.0
            speed = {}
            boxes = []
            if model is not None:
                result = model.predict(frame, **predict_args)[0]
                boxes = result.boxes.data.cpu().numpy()
                if args.device != "cpu":
                    torch.cuda.synchronize(model.predictor.device)
                predict_ms = (time.perf_counter() - infer_start) * 1000
                speed = result.speed
                last_result = result
            ready = time.perf_counter()
            if not args.no_display:
                shown = result.plot() if model is not None else frame.copy()
                fps = (len(samples) + 1) / max(ready - started, 1e-9)
                cv2.putText(shown, f"FPS {fps:.1f}  PRED {predict_ms:.1f}ms  AGE {age_ms:.1f}ms",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow("D435i Detection (Q/ESC quit)", shown)
                key = cv2.waitKey(1) & 0xff
            else:
                key = -1
            finished = time.perf_counter()
            samples.append({"predict_ms": predict_ms, "age_ms": age_ms,
                            "local_result_latency_ms": (ready - captured_at) * 1000,
                            "loop_ms": (finished - tick) * 1000,
                            "display_ms": (finished - ready) * 1000,
                            "detections": len(boxes), **{k + "_ms": v for k, v in speed.items()}})
            last_frame = frame
            if finished - last_log >= 2:
                recent = samples[-30:]
                print(f"[STATS] FPS {len(samples)/(finished-started):.2f} | "
                      f"PRED {np.mean([s['predict_ms'] for s in recent]):.2f} ms | "
                      f"AGE {age_ms:.2f} ms | boxes {len(boxes)} | skipped {skipped}", flush=True)
                last_log = finished
            if key in (ord("q"), 27):
                interrupted = True
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        elapsed = time.perf_counter() - started if started is not None else 0
        received, sdk_gaps = capture.counters() if capture is not None else (0, 0)
        if capture is not None:
            capture.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if not samples:
        raise RuntimeError("未完成有效测试帧，不能生成性能结论")
    summary = {
        "camera": capture.info, "model": str(Path(args.model).expanduser().resolve()),
        "model_type": args.model_type, "imgsz": args.imgsz, "confidence": args.conf,
        "device": args.device, "capture_only": args.capture_only,
        "display": not args.no_display, "warmup_frames": args.warmup,
        "elapsed_seconds": elapsed, "interrupted": interrupted, "processed_frames": len(samples),
        "processed_fps": len(samples) / elapsed,
        "captured_frames": received - initial_received,
        "capture_fps": (received - initial_received) / elapsed,
        "skipped_by_consumer": skipped, "sdk_frame_number_gaps": sdk_gaps - initial_gaps,
        "metrics": {key: describe(np, [s[key] for s in samples]) for key in samples[0]},
        "latency_scope": "SDK receive to local result ready; excludes exposure/USB/SDK buffering",
        "software": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                     "torch": torch.__version__ if torch is not None else None},
    }
    if model is not None:
        summary["class_names"] = last_result.names
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "frames.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    snapshot = last_result.plot() if model is not None else last_frame
    if not cv2.imwrite(str(output / "last_frame.jpg"), snapshot):
        raise OSError("保存测试画面失败")
    print(f"[RESULT] {len(samples)} 帧 / {elapsed:.2f}s = {summary['processed_fps']:.2f} FPS", flush=True)
    for key, stats in summary["metrics"].items():
        if key.endswith("_ms"):
            print(f"[RESULT] {key}: mean={stats['mean']:.2f}, p95={stats['p95']:.2f} ms")
    print(f"[RESULT] 输出: {output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(benchmark(parse_args()))
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
