#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noFFN (SO-DETR-R18-NoFFT) 单张图片 / 视频检测测试脚本
适配 RT-DETR 架构与 VisDrone 10 类航拍模型。
"""

import sys
import os
import argparse
from pathlib import Path

# 确保加载 SO_DETR 内置的定制 ultralytics 库
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
SO_DETR_DIR = WORKSPACE_DIR / "SO_DETR"
if str(SO_DETR_DIR) not in sys.path:
    sys.path.insert(0, str(SO_DETR_DIR))

try:
    from ultralytics import RTDETR
except ImportError as e:
    print(f"[错误] 无法从 SO_DETR 导入 RTDETR: {e}")
    print("请确认 SO_DETR 目录完整。")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="noFFN RT-DETR 目标检测推理测试")
    parser.add_argument(
        "--model",
        type=str,
        default=str(WORKSPACE_DIR / "weights" / "noFFN_fp16.engine"),
        help="模型权重路径 (.engine / .pt / .onnx)，默认: weights/noFFN_fp16.engine",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(WORKSPACE_DIR / "SO_DETR" / "runs/train/Q0/train_batch0.jpg"),
        help="输入图像/视频路径或 RTSP 地址，默认测试自带航拍样本",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值 (默认 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理尺寸 (默认 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备 (0 为 GPU, cpu 为 CPU)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(WORKSPACE_DIR / "test_results"),
        help="结果图片保存目录 (默认 test_results)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    # 原样保留 source 字符串：Path 会把 rtsp:// 的 "//" 折叠成 "/"，破坏 RTSP 地址
    source = args.source
    is_rtsp = source.startswith("rtsp://")
    source_path = Path(source)

    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        print("可用模型推荐: noFFN_fp16.engine, noFFN.engine, noFFN.pt")
        return

    if not is_rtsp and not source_path.exists():
        print(f"[错误] 输入源不存在: {source_path}")
        return

    print("=" * 60)
    print("🚀 正在加载 noFFN RT-DETR 模型...")
    print(f"📦 模型路径: {model_path}")
    print(f"🖼️  输入源:   {source}")
    print(f"⚙️  置信度:   {args.conf} | 尺寸: {args.imgsz} | 设备: {args.device}")
    print("=" * 60)

    model = RTDETR(str(model_path))

    print("\n🔍 开始推理...")
    results = model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=args.save_dir,
        name="predict",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("📊 检测结果统计:")
    for idx, r in enumerate(results):
        boxes = r.boxes
        total_boxes = len(boxes)
        print(f"[{idx+1}] 输入: {r.path} -> 共检出 {total_boxes} 个目标")

        # 统计各类别检测数量
        class_counts = {}
        for c in boxes.cls.tolist():
            c_id = int(c)
            c_name = r.names.get(c_id, f"class_{c_id}")
            class_counts[c_name] = class_counts.get(c_name, 0) + 1

        if class_counts:
            print("   各类别分布:")
            for c_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"     - {c_name:<16}: {count} 个")
        else:
            print("   (当前置信度阈值下未检出目标，可适当调低 --conf 比如 0.1)")

    save_path = Path(args.save_dir) / "predict"
    print("=" * 60)
    print(f"💾 标注结果图片已保存至: {save_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
