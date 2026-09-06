"""无需相机/GPU：真实 spawn 进程中的断流、阻塞与故障切换测试。"""

import contextlib
import io
import multiprocessing as mp
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import rtsp_capture as rtsp
import A8mini_RTSP_YOLO_Detection as app


class FakeCapture:
    def __init__(self, source, backend):
        self.source, self.backend, self.count = source, backend, 0

    def isOpened(self):
        return True

    def read(self):
        if self.source == "hang-read" and self.backend == "gstreamer":
            time.sleep(60)
        if self.source == "hang-release" and self.backend == "gstreamer":
            return False, None
        if self.source == "flaky" and self.backend == "gstreamer" and self.count:
            return False, None
        time.sleep(0.01)
        self.count += 1
        return True, np.full((2, 2, 3), self.count % 256, dtype=np.uint8)

    def release(self):
        if self.source == "hang-release" and self.backend == "gstreamer":
            time.sleep(60)


def fake_open(config, backend):
    if config.source == "hang-open" and backend == "gstreamer":
        time.sleep(60)
    if config.source == "error-open" and backend == "gstreamer":
        raise RuntimeError("injected open error")
    return FakeCapture(config.source, backend)


def fake_worker(config, backend, slot, connection):
    rtsp.open_capture = fake_open
    rtsp.capture_worker(config, backend, slot, connection)


class CaptureTests(unittest.TestCase):
    def config(self, source, **kwargs):
        return rtsp.CaptureConfig(source, open_timeout_ms=80, read_timeout_ms=80,
                                  reconnect_delay=0.01, max_pixels=4, **kwargs)

    def wait_packet(self, capture, backend=None, previous=None, timeout=8):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            packet = capture.read_latest(previous)
            if packet is not None and (backend is None or packet.backend == backend):
                return packet
            time.sleep(0.01)
        self.fail(f"没有收到 {backend} 的新帧: {capture.status}")

    def test_auto_recovers_from_native_hangs_and_errors(self):
        for source in ("hang-open", "hang-read", "hang-release", "error-open", "flaky"):
            with self.subTest(source=source):
                before = {p.pid for p in mp.active_children()}
                capture = rtsp.LatestFrameCapture(self.config(source), worker=fake_worker).start()
                try:
                    packet = self.wait_packet(capture, "ffmpeg")
                    self.assertEqual(packet.sequence[0], 3)
                    self.assertEqual(packet.frame.shape, (2, 2, 3))
                finally:
                    capture.close()
                self.assertEqual({p.pid for p in mp.active_children()}, before)

    def test_explicit_backend_reconnects_without_switching(self):
        capture = rtsp.LatestFrameCapture(
            self.config("flaky", backend="gstreamer"), worker=fake_worker).start()
        try:
            first = self.wait_packet(capture)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                packet = self.wait_packet(capture, previous=first.sequence)
                self.assertEqual(packet.backend, "gstreamer")
                if packet.sequence[0] > first.sequence[0]:
                    break
            else:
                self.fail("未重新连接")
        finally:
            capture.close()

    def test_slow_consumer_skips_frames_and_owns_snapshot(self):
        capture = rtsp.LatestFrameCapture(
            self.config("normal", backend="ffmpeg"), worker=fake_worker).start()
        try:
            first = self.wait_packet(capture)
            original = first.frame.copy()
            time.sleep(0.15)  # 模拟耗时推理；采集继续推进。
            second = self.wait_packet(capture, previous=first.sequence)
            self.assertGreater(second.sequence[1] - first.sequence[1], 3)
            np.testing.assert_array_equal(first.frame, original)
        finally:
            capture.close()

    def test_close_during_blocked_read_is_bounded(self):
        capture = rtsp.LatestFrameCapture(
            self.config("hang-read", backend="gstreamer"), worker=fake_worker).start()
        time.sleep(0.5)
        started = time.monotonic()
        capture.close()
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(capture._thread.is_alive())

    def test_shared_slot_rejects_duplicates_and_stale_frames(self):
        slot = rtsp.SharedFrame(mp.get_context("spawn"), 4)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        slot.publish(frame, time.monotonic(), 12)
        snapshot = slot.take(0, 1000)
        self.assertEqual(snapshot[3], 12)
        self.assertIsNone(slot.take(snapshot[0], 1000))
        snapshot[1][:] = 255
        np.testing.assert_array_equal(slot.take(0, 1000)[1], frame)
        slot.publish(frame, time.monotonic() - 2, 12)
        self.assertIsNone(slot.take(0, 1000))

    def test_killed_worker_lock_cannot_block_consumer(self):
        slot = rtsp.SharedFrame(mp.get_context("spawn"), 4)
        slot.lock.acquire()
        try:
            started = time.monotonic()
            self.assertIsNone(slot.take(0, 1000))
            self.assertLess(time.monotonic() - started, 0.1)
        finally:
            slot.lock.release()

    def test_capacity_checked(self):
        slot = rtsp.SharedFrame(mp.get_context("spawn"), 4)
        with self.assertRaisesRegex(ValueError, "容量"):
            slot.publish(np.zeros((3, 3, 3), dtype=np.uint8), time.monotonic(), 1)

    def test_both_backends_receive_unclamped_open_timeouts(self):
        config = rtsp.CaptureConfig("rtsp://camera/main.264", read_timeout_ms=800)
        for backend in ("gstreamer", "ffmpeg"):
            with patch.object(rtsp.cv2, "VideoCapture") as constructor, patch.dict(rtsp.os.environ):
                rtsp.open_capture(config, backend)
                params = constructor.call_args.args[2]
                self.assertEqual(params, [rtsp.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000,
                                          rtsp.cv2.CAP_PROP_READ_TIMEOUT_MSEC, 800])

    def test_codecs(self):
        for codec in ("h264", "h265"):
            pipeline = rtsp.build_gstreamer_pipeline('rtsp://camera/"stream', 100, 800, codec)
            self.assertIn(f"rtp{codec}depay ! {codec}parse", pipeline)
            self.assertIn("tcp-timeout=800000", pipeline)
            self.assertIn('\\"stream', pipeline)
            self.assertIn("max-buffers=1", pipeline)

    def test_cli_defaults_and_invalid_values(self):
        defaults = app.parse_args([])
        self.assertEqual(defaults.read_timeout, 1000)
        self.assertFalse(defaults.save)
        self.assertTrue(defaults.source.endswith("main.264"))
        for argv in (["--read-timeout", "0"], ["--backend-failures", "0"],
                     ["--save-crops"], ["--reconnect-delay", "nan"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    app.parse_args(argv)

    def test_save_full_and_unannotated_crop(self):
        raw = np.full((10, 10, 3), 30, dtype=np.uint8)
        annotated = np.full((10, 10, 3), 200, dtype=np.uint8)
        detection = {"box": [-2, 1, 5, 6], "class_name": "person", "confidence": 0.9}
        with tempfile.TemporaryDirectory() as directory:
            app.save_detection_result(annotated, raw, [detection], Path(directory), True)
            files = list(Path(directory).glob("*.jpg"))
            self.assertEqual(len(files), 2)
            crop_path = next(p for p in files if "full" not in p.name)
            crop = rtsp.cv2.imread(str(crop_path))
            self.assertEqual(crop.shape, (5, 5, 3))
            self.assertLess(abs(float(crop.mean()) - 30), 2)

    def test_main_prediction_and_optional_saving(self):
        for save in (False, True):
            with self.subTest(save=save), tempfile.TemporaryDirectory() as directory:
                # 七列跟踪格式也应读取末两列 conf/cls，不能把 track ID 当 conf。
                tensor = unittest.mock.Mock()
                tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(
                    [[1, 1, 5, 6, 99, 0.9, 0]], dtype=np.float32)
                result = types.SimpleNamespace(
                    boxes=types.SimpleNamespace(data=tensor), names={0: "person"})
                model = unittest.mock.Mock()
                model.predict.return_value = [result]
                packet = rtsp.FramePacket(np.full((10, 10, 3), 30, dtype=np.uint8),
                                          (1, 1), time.monotonic(), 12, "ffmpeg")
                capture = unittest.mock.Mock()
                capture.start.return_value = capture
                capture.read_latest.side_effect = [packet, KeyboardInterrupt]
                argv = ["app", "--model", "dummy.pt", "--no-display",
                        "--save-dir", str(Path(directory) / "images")]
                if save:
                    argv += ["--save", "--save-crops"]
                with patch.object(sys, "argv", argv), patch.dict(
                    sys.modules, {"ultralytics": types.SimpleNamespace(YOLO=lambda *a, **k: model)}
                ), patch.object(app, "LatestFrameCapture", return_value=capture), patch.object(
                    app, "save_detection_result", wraps=app.save_detection_result
                ) as saver, patch.object(app.cv2, "destroyAllWindows") as destroy:
                    self.assertEqual(app.main(), 0)
                    self.assertFalse(model.predict.call_args.kwargs["rect"])
                    tensor.detach.return_value.cpu.assert_called_once()
                    capture.close.assert_called_once()
                    destroy.assert_not_called()
                    if save:
                        saver.assert_called_once()
                        self.assertEqual(saver.call_args.args[2][0]["class_id"], 0)
                        self.assertAlmostEqual(saver.call_args.args[2][0]["confidence"], 0.9)
                        self.assertEqual(float(saver.call_args.args[1].mean()), 30)
                    else:
                        saver.assert_not_called()
                        self.assertFalse((Path(directory) / "images").exists())


if __name__ == "__main__":
    unittest.main()
