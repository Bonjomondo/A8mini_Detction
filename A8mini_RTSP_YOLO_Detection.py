# -*- coding: utf-8 -*-
"""
A8 mini RTSP + Ultralytics YOLO/TensorRT 实时检测 V1.2

功能：
1. 从 A8 mini 网口 RTSP 视频流读取画面；
2. 使用 .engine 或 .pt 模型进行目标检测；
3. 显示检测框、类别、置信度和处理 FPS；
4. 使用 --save 启用按类别保存完整画面（默认关闭）；
5. 可选保存目标裁剪图；
6. 最新帧采集、进程 watchdog、运行中自动切换后端；
7. 显示 CAP / YOLO / AGE 分项耗时。

A8 mini 常用旧版 RTSP 地址：
    rtsp://192.168.144.25:8554/main.264

使用示例：
    python3 A8mini_RTSP_YOLO_Detection.py

    python3 A8mini_RTSP_YOLO_Detection.py \
        --model yolo11s.engine \
        --names-model yolo11s.pt \
        --save --save-classes person \
        --save-crops
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from rtsp_capture import CaptureConfig, LatestFrameCapture


def get_class_name(names, cls_id: int) -> str:
    """根据类别编号获取类别名称。"""
    if isinstance(names, dict):
        return names.get(cls_id, names.get(str(cls_id), f"class{cls_id}"))

    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return names[cls_id]

    return f"class{cls_id}"


def draw_detection(frame, box, cls_id: int, conf: float, names) -> None:
    """在画面上绘制检测框、类别名称和置信度。"""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))

    cls_name = get_class_name(names, cls_id)
    label = f"{cls_name} {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    label_bottom = y1 - 5
    if label_bottom - text_height - baseline < 0:
        label_bottom = y1 + text_height + baseline + 5

    label_left = x1
    label_right = min(label_left + text_width + 6, frame_width - 1)
    label_top = max(0, label_bottom - text_height - baseline - 6)

    cv2.rectangle(
        frame,
        (label_left, label_top),
        (label_right, label_bottom + 3),
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (label_left + 3, label_bottom - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def load_class_names(names_model_path: Optional[str]):
    """从原始 .pt 模型中读取类别名称。"""
    if not names_model_path:
        return None

    path = Path(names_model_path).expanduser()
    if not path.exists():
        print(f"[WARN] 类别名称模型不存在: {path}")
        print("[WARN] 将使用推理模型自带的类别名称。")
        return None

    print(f"[INFO] Loading class names from: {path}")
    from ultralytics import YOLO

    names_model = YOLO(str(path), task="detect")
    names = names_model.names
    del names_model

    if names is not None:
        print(f"[INFO] Loaded {len(names)} class names.")

    return names


def parse_save_classes(save_classes_text: str):
    """解析需要触发保存的类别名称或类别编号。"""
    target_names = set()
    target_ids = set()
    save_all = False

    if not save_classes_text:
        return target_names, target_ids, save_all

    for item in save_classes_text.split(","):
        item = item.strip()
        if not item:
            continue

        if item.lower() == "all":
            save_all = True
            continue

        try:
            target_ids.add(int(item))
        except ValueError:
            target_names.add(item.lower())

    return target_names, target_ids, save_all


def is_target_class(
    cls_id: int,
    cls_name: str,
    target_names: set,
    target_ids: set,
    save_all: bool,
) -> bool:
    """判断当前类别是否需要触发保存。"""
    return (
        save_all
        or cls_id in target_ids
        or cls_name.lower() in target_names
    )


def safe_filename(text: str) -> str:
    """将类别名称转换成适合放入文件名的字符串。"""
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")
    return text if text else "object"


def save_detection_result(
    annotated_frame,
    original_frame,
    target_detections,
    save_dir: Path,
    save_crops: bool,
) -> None:
    """保存完整检测画面，并可选保存目标裁剪图。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    class_names = sorted(
        {safe_filename(item["class_name"]) for item in target_detections}
    )
    class_text = "_".join(class_names)

    full_image_path = save_dir / f"{timestamp}_{class_text}_full.jpg"
    if cv2.imwrite(str(full_image_path), annotated_frame):
        print(f"[SAVE] 完整画面: {full_image_path}")
    else:
        print(f"[ERROR] 完整画面保存失败: {full_image_path}")

    if not save_crops:
        return

    frame_height, frame_width = original_frame.shape[:2]

    for index, detection in enumerate(target_detections, start=1):
        x1, y1, x2, y2 = map(int, detection["box"])
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = original_frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        class_name = safe_filename(detection["class_name"])
        confidence = detection["confidence"]
        crop_path = save_dir / (
            f"{timestamp}_{class_name}_{confidence:.2f}_{index:02d}.jpg"
        )

        if cv2.imwrite(str(crop_path), crop):
            print(f"[SAVE] 目标裁剪图: {crop_path}")
        else:
            print(f"[ERROR] 裁剪图保存失败: {crop_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="A8 mini RTSP YOLO V1.2 latest-frame detection"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.engine",
        help="推理模型，例如 yolo11s.engine 或 yolo11s.pt",
    )
    parser.add_argument(
        "--names-model",
        type=str,
        default="",
        help="可选：用于读取类别名称的原始 PT 模型；默认直接使用 engine 内置名称",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="rtsp://192.168.144.25:8554/main.264",
        help="A8 mini RTSP 地址",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "ffmpeg", "gstreamer"],
        default="auto",
        help="auto 优先 GStreamer，连续连接或读帧失败后切换后端",
    )
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264",
                        help="GStreamer 解码格式，必须匹配相机实际输出")
    parser.add_argument("--backend-failures", type=int, default=2,
                        help="auto 连续失败多少次后切换后端")
    parser.add_argument("--max-frame-age", type=int, default=1000,
                        help="只处理解码完成后不超过此帧龄的帧，单位毫秒")
    parser.add_argument("--capture-max-pixels", type=int, default=3840 * 2160,
                        help="共享帧槽最大像素数，默认支持 3840x2160")
    parser.add_argument("--stats-interval", type=float, default=2.0,
                        help="终端分项耗时输出间隔（秒），0 关闭")
    parser.add_argument("--save", action="store_true",
                        help="启用按类别保存图片，默认关闭")
    parser.add_argument(
        "--latency",
        type=int,
        default=150,
        help="GStreamer RTSP 缓冲延时，单位毫秒",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA GPU 编号，Orin NX 通常为 0",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="置信度阈值",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU 阈值",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO 推理尺寸，使用固定尺寸 engine 时必须匹配 engine",
    )
    parser.add_argument(
        "--save-classes",
        type=str,
        default="person",
        help="触发保存的类别，例如 person、person,car、0,2 或 all",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="a8mini_detection_images",
        help="检测图片保存目录",
    )
    parser.add_argument(
        "--save-cooldown",
        type=float,
        default=2.0,
        help="两次保存之间的最短时间，单位秒；0 表示每帧均可保存",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="同时保存每个检测目标的裁剪图",
    )
    parser.add_argument(
        "--max-read-failures",
        type=int,
        default=1,
        help="连续读帧失败多少次后重连；读帧已有超时，建议保持 1",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=0.2,
        help="RTSP 重连等待时间，单位秒",
    )
    parser.add_argument(
        "--open-timeout",
        type=int,
        default=2000,
        help="RTSP 打开超时，单位毫秒",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=1000,
        help="RTSP 单次读帧超时，单位毫秒；超时后进入重连",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="不弹出 OpenCV 窗口，适合无桌面环境运行",
    )

    args = parser.parse_args(argv)
    for name in ("open_timeout", "read_timeout", "max_read_failures",
                 "backend_failures", "max_frame_age", "capture_max_pixels", "imgsz"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于 0")
    for name in ("latency", "reconnect_delay", "save_cooldown", "stats_interval"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} 必须是非负有限数值")
    for name in ("conf", "iou"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} 必须在 0 到 1 之间")
    if args.save_crops and not args.save:
        parser.error("--save-crops 需要同时指定 --save")
    return args


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    if model_path.suffix.lower() == ".engine" and not model_path.exists():
        print(f"[ERROR] TensorRT engine 不存在: {model_path}")
        return 1

    save_dir = Path(args.save_dir).expanduser()
    if args.save:
        save_dir.mkdir(parents=True, exist_ok=True)
    target_names, target_ids, save_all = parse_save_classes(args.save_classes)
    print(f"[INFO] RTSP source: {args.source}")
    print(f"[INFO] Backend: {args.backend}, codec: {args.codec}")
    print(f"[INFO] Model: {model_path}, imgsz: {args.imgsz}, GPU: {args.gpu}")
    print(f"[INFO] RTSP open/read timeout: {args.open_timeout}/{args.read_timeout} ms")
    print(f"[INFO] 图片保存: {save_dir.resolve() if args.save else '关闭（--save 启用）'}")

    capture = None
    exit_code = 0
    try:
        # spawn 采集子进程不会导入 Ultralytics 或初始化 CUDA。
        from ultralytics import YOLO

        model = YOLO(str(model_path), task="detect")
        names = load_class_names(args.names_model)
        capture = LatestFrameCapture(CaptureConfig(
            source=args.source, backend=args.backend, codec=args.codec,
            latency_ms=args.latency, open_timeout_ms=args.open_timeout,
            read_timeout_ms=args.read_timeout, reconnect_delay=args.reconnect_delay,
            max_read_failures=args.max_read_failures, backend_failures=args.backend_failures,
            max_frame_age_ms=args.max_frame_age, max_pixels=args.capture_max_pixels,
        )).start()
        previous_sequence = None
        previous_time = None
        fps_smooth = 0.0
        last_save_time = -float("inf")
        last_stats_time = 0.0
        last_idle_display = 0.0
        window = "A8 mini RTSP YOLO Detection"
        if not args.no_display:
            print("[INFO] Press Q or ESC to quit.")

        while True:
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            packet = capture.read_latest(previous_sequence)
            if packet is None:
                now = time.monotonic()
                if previous_time is not None and now - previous_time > args.max_frame_age / 1000:
                    previous_time = None
                    fps_smooth = 0.0
                if not args.no_display and now - last_idle_display >= 0.2:
                    # 不把旧检测画面伪装成实时结果。
                    waiting = np.zeros((180, 800, 3), dtype=np.uint8)
                    cv2.putText(waiting, f"Waiting for fresh frame: {capture.status}",
                                (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 200, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window, waiting)
                    last_idle_display = now
                if args.stats_interval and now - last_stats_time >= args.stats_interval:
                    print(f"[STATS] RTSP: {capture.status}; waiting for fresh frame", flush=True)
                    last_stats_time = now
                time.sleep(0.01)
                continue

            if previous_sequence is None or packet.sequence[0] != previous_sequence[0]:
                previous_time = None
                fps_smooth = 0.0
            previous_sequence = packet.sequence
            frame = packet.frame
            started = time.monotonic()
            age_ms = (started - packet.captured_at) * 1000
            results = model.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                device=args.gpu, rect=False, verbose=False,
            )
            result = results[0]
            # Boxes 为 6 列检测或 7 列跟踪数据；置信度/类别始终是末两列。
            box_data = (result.boxes.data.detach().cpu().numpy()
                        if result.boxes is not None else [])
            yolo_ms = (time.monotonic() - started) * 1000
            current_names = names if names is not None else result.names
            target_detections = []
            for row in box_data:
                cls_id, confidence = int(row[-1]), float(row[-2])
                cls_name = get_class_name(current_names, cls_id)
                if args.save and is_target_class(cls_id, cls_name, target_names, target_ids, save_all):
                    target_detections.append({
                        "box": row[:4].copy(), "class_id": cls_id,
                        "class_name": cls_name, "confidence": confidence,
                    })
            should_save = (bool(target_detections) and
                           time.monotonic() - last_save_time >= args.save_cooldown)
            original_frame = frame.copy() if should_save and args.save_crops else None
            for row in box_data:
                draw_detection(frame, row[:4], int(row[-1]), float(row[-2]), current_names)

            now = time.monotonic()
            if previous_time is not None and now > previous_time:
                fps = 1.0 / (now - previous_time)
                fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps
            previous_time = now
            metrics = (f"CAP: {packet.cap_ms:.1f} ms  YOLO: {yolo_ms:.1f} ms  "
                       f"AGE: {age_ms:.1f} ms")
            for y, text in ((35, f"FPS: {fps_smooth:.1f}  RTSP: {packet.backend}"), (65, metrics)):
                cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 255), 2, cv2.LINE_AA)

            save_ms = 0.0
            if should_save:
                save_started = time.monotonic()
                save_detection_result(frame, original_frame, target_detections, save_dir, args.save_crops)
                last_save_time = time.monotonic()
                save_ms = (last_save_time - save_started) * 1000
            if args.stats_interval and now - last_stats_time >= args.stats_interval:
                print(f"[STATS] {packet.backend} FPS: {fps_smooth:.1f}  {metrics}  SAVE: {save_ms:.1f} ms",
                      flush=True)
                last_stats_time = now
            if not args.no_display:
                cv2.imshow(window, frame)
                last_idle_display = time.monotonic()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received.")
    except Exception as error:
        exit_code = 1
        print(f"\n[ERROR] {type(error).__name__}: {error}")
        if isinstance(error, AssertionError):
            print(f"[INFO] 请检查 --imgsz {args.imgsz} 是否匹配 engine；具体原因见上方错误。")
    finally:
        if capture is not None:
            capture.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("[INFO] Stopped.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
