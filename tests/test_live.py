import unittest

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover
    cv2 = None

from yolostudio import live
from yolostudio.counting import Roi


class _Tensor(list):
    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, rows):
        self.xyxy = _Tensor([r[:4] for r in rows])
        self.conf = _Tensor([r[4] for r in rows])
        self.cls = _Tensor([r[5] for r in rows])
        self.id = None

    def __len__(self):
        return len(self.cls)


class _Result:
    names = {0: "person", 1: "car"}

    def __init__(self, rows):
        self.boxes = _Boxes(rows)


class FakeModel:
    names = {0: "person", 1: "car"}

    def __init__(self, rows):
        self.rows = rows

    def predict(self, *args, **kwargs):
        return [_Result(self.rows)]

    __call__ = predict


@unittest.skipIf(cv2 is None, "opencv/numpy not installed")
class LiveTests(unittest.TestCase):
    frame = None

    def setUp(self):
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        # one person with its centre inside the default ROI, one car with its centre outside it
        self.model = FakeModel([(80, 30, 120, 70, 0.9, 0), (0, 0, 10, 10, 0.8, 1)])

    def test_analyse_frame_counts_total_and_roi(self):
        rgb, detections, total, in_roi = live.analyse_frame(self.model, self.frame, 0.25, None, Roi(), True)
        self.assertEqual(rgb.shape, self.frame.shape)
        self.assertEqual(dict(total), {"person": 1, "car": 1})
        self.assertEqual(dict(in_roi), {"person": 1})

    def test_session_records_counts_and_returns_bgr(self):
        session = live.LiveSession()
        session.configure(self.model, 0.25, None, Roi(), True)
        out = session.process(self.frame)
        self.assertEqual(out.shape, self.frame.shape)
        total, in_roi, frames = session.snapshot()
        self.assertEqual((sum(total.values()), sum(in_roi.values()), frames), (2, 1, 1))
        session.reset()
        self.assertEqual(session.snapshot()[2], 0)

    def test_weights_name_is_loaded_lazily_once(self):
        calls = []
        original = live.inference.load_model
        live.inference.load_model = lambda name: calls.append(name) or self.model
        try:
            live._MODELS.clear()
            session = live.LiveSession()
            session.configure("fake.pt", 0.25, None, Roi(), True)
            self.assertEqual(calls, [])
            session.process(self.frame)
            session.process(self.frame)
            self.assertEqual(calls, ["fake.pt"])
        finally:
            live.inference.load_model = original
            live._MODELS.clear()

    def test_session_without_a_model_passes_frames_through(self):
        session = live.LiveSession()
        self.assertIs(session.process(self.frame), self.frame)


if __name__ == "__main__":
    unittest.main()
