"""Sampling tests run without inference dependencies; real codec test is optional."""

import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from video2.video import sample_video


class FakeFrame:
    def __init__(self, height=100, width=200):
        self.shape = (height, width, 3)

    def copy(self):
        return FakeFrame(*self.shape[:2])


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "clip.avi"
        self.path.touch()
        self.capture = Mock()
        self.capture.isOpened.return_value = True
        self.capture.get.side_effect = lambda prop: {1: 10.0, 2: 11.0}[prop]
        self.capture.set.return_value = True
        self.capture.read.side_effect = lambda: (True, FakeFrame())
        cv2 = types.SimpleNamespace(
            CAP_PROP_FPS=1,
            CAP_PROP_FRAME_COUNT=2,
            CAP_PROP_POS_FRAMES=3,
            INTER_AREA=4,
            COLOR_BGR2RGB=5,
            VideoCapture=Mock(return_value=self.capture),
            resize=Mock(side_effect=lambda frame, size, **kw: FakeFrame(size[1], size[0])),
            cvtColor=Mock(side_effect=lambda frame, mode: frame.copy()),
        )
        self.cv2 = cv2
        modules = patch.dict("sys.modules", {"cv2": cv2, "numpy": types.ModuleType("numpy")})
        modules.start()
        self.addCleanup(modules.stop)

    def test_uniform_sampling_includes_endpoints_and_rgb_metadata(self):
        result = sample_video(self.path, num_frames=4, max_frame_size=100)
        self.assertEqual(result.timestamps, [0.0, 0.3, 0.7, 1.0])
        self.assertEqual(result.duration_seconds, 1.1)
        self.assertEqual(result.source_fps, 10.0)
        self.assertEqual(result.effective_fps, 3.0)
        self.assertEqual(len(result.frames), 4)
        self.assertEqual(result.frames[0].shape, (50, 100, 3))
        self.assertEqual([c.args[1] for c in self.capture.set.call_args_list], [0, 3, 7, 10])
        self.assertEqual(self.cv2.cvtColor.call_args.args[1], self.cv2.COLOR_BGR2RGB)
        self.capture.release.assert_called_once()

    def test_short_video_duplicates_frames_with_source_timestamps(self):
        self.capture.get.side_effect = lambda prop: {1: 10.0, 2: 2.0}[prop]
        result = sample_video(self.path, num_frames=4)
        self.assertEqual(result.timestamps, [0.0, 0.0, 0.1, 0.1])
        self.assertEqual(self.capture.read.call_count, 2)
        self.assertIsNot(result.frames[0], result.frames[1])
        self.assertEqual(result.frames[0].shape, (100, 200, 3))

    def test_single_frame_uses_source_fps(self):
        self.capture.get.side_effect = lambda prop: {1: 10.0, 2: 1.0}[prop]
        result = sample_video(self.path, num_frames=4)
        self.assertEqual(result.timestamps, [0.0] * 4)
        self.assertEqual(result.effective_fps, 10.0)
        self.assertEqual(self.capture.read.call_count, 1)

    def test_invalid_options(self):
        for value in (0, 1, 3, True, 2.0):
            with self.subTest(num_frames=value), self.assertRaises(ValueError):
                sample_video(self.path, num_frames=value)
        for value in (0, 27, True, 100.0):
            with self.subTest(max_frame_size=value), self.assertRaises(ValueError):
                sample_video(self.path, max_frame_size=value)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            sample_video(self.path.with_name("missing.avi"))

    def test_invalid_metadata_releases_capture(self):
        for fps, count in ((0, 10), (float("nan"), 10), (10, 0), (10, float("inf"))):
            self.capture.get.side_effect = lambda prop, f=fps, n=count: {1: f, 2: n}[prop]
            with self.subTest(fps=fps, count=count), self.assertRaises(ValueError):
                sample_video(self.path)
        self.assertEqual(self.capture.release.call_count, 4)

    def test_decode_failure_is_not_skipped(self):
        self.capture.read.side_effect = [(True, FakeFrame()), (False, None)]
        with self.assertRaisesRegex(RuntimeError, "Cannot decode.*frame 10"):
            sample_video(self.path, num_frames=2)
        self.capture.release.assert_called_once()

    def test_seek_failure(self):
        self.capture.set.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Cannot seek"):
            sample_video(self.path, num_frames=2)
        self.capture.release.assert_called_once()

    def test_open_failure(self):
        self.capture.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Cannot open"):
            sample_video(self.path)
        self.capture.release.assert_called_once()

    def test_missing_dependencies(self):
        with patch.dict("sys.modules", {"cv2": None}):
            with self.assertRaisesRegex(RuntimeError, "pip install"):
                sample_video(self.path)


class RealCodecTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("cv2") is not None and importlib.util.find_spec("numpy") is not None,
        "optional OpenCV/NumPy video dependencies not installed",
    )
    def test_decode_generated_video(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clip.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 40))
            self.assertTrue(writer.isOpened(), "MJPG codec unavailable")
            try:
                for index in range(10):
                    frame = np.zeros((40, 80, 3), dtype=np.uint8)
                    frame[:, :, 2] = index * 20
                    writer.write(frame)
            finally:
                writer.release()
            result = sample_video(path, num_frames=4, max_frame_size=56)
            self.assertEqual(result.timestamps, [0.0, 0.3, 0.6, 0.9])
            self.assertEqual([f.shape for f in result.frames], [(28, 56, 3)] * 4)
            self.assertGreater(result.frames[-1][:, :, 0].mean(), 170)
            self.assertLess(result.frames[-1][:, :, 2].mean(), 10)


if __name__ == "__main__":
    unittest.main()
