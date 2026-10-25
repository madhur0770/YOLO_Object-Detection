import importlib.util
import unittest

from yolostudio.counting import Detection, Roi
from yolostudio.inference import detect, results_to_detections


class Listable:
    """Stands in for a tensor: the code under test only calls .tolist()."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class FakeBoxes:
    def __init__(self, xyxy, conf, cls, ids=None):
        self.xyxy, self.conf, self.cls = Listable(xyxy), Listable(conf), Listable(cls)
        self.id = Listable(ids) if ids is not None else None


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    names = {0: "mask", 1: "no-mask"}

    def __init__(self, result):
        self.result, self.calls = result, []

    def __call__(self, frame, **kwargs):
        self.calls.append(("predict", kwargs))
        return [self.result]

    def track(self, frame, **kwargs):
        self.calls.append(("track", kwargs))
        return [self.result]


BOXES = FakeBoxes([[10, 20, 110, 220], [0, 0, 50, 50]], [0.9, 0.4], [0, 1], ids=[5, 6])


class ConversionTests(unittest.TestCase):
    def test_converts_boxes_to_detections(self):
        dets = results_to_detections(FakeResult(BOXES), FakeModel.names)
        self.assertEqual(dets[0], Detection("mask", 0.9, 10.0, 20.0, 110.0, 220.0, 5))
        self.assertEqual(dets[1].label, "no-mask")
        self.assertEqual(dets[1].track_id, 6)

    def test_no_track_ids_when_absent(self):
        boxes = FakeBoxes([[0, 0, 5, 5]], [0.5], [0])
        self.assertIsNone(results_to_detections(FakeResult(boxes), FakeModel.names)[0].track_id)

    def test_class_filter(self):
        dets = results_to_detections(FakeResult(BOXES), FakeModel.names, allowed={"no-mask"})
        self.assertEqual([d.label for d in dets], ["no-mask"])

    def test_empty_or_missing_boxes(self):
        self.assertEqual(results_to_detections(FakeResult(FakeBoxes([], [], [])), FakeModel.names), [])
        self.assertEqual(results_to_detections(FakeResult(None), FakeModel.names), [])


class DetectTests(unittest.TestCase):
    def test_predict_passes_confidence(self):
        model = FakeModel(FakeResult(BOXES))
        detect(model, object(), confidence=0.3)
        self.assertEqual(model.calls[0][0], "predict")
        self.assertEqual(model.calls[0][1]["conf"], 0.3)

    def test_track_mode_uses_the_tracker(self):
        model = FakeModel(FakeResult(BOXES))
        detect(model, object(), track=True)
        self.assertEqual(model.calls[0][0], "track")
        self.assertTrue(model.calls[0][1]["persist"])


@unittest.skipIf(importlib.util.find_spec("cv2") is None or importlib.util.find_spec("numpy") is None, "OpenCV not installed")
class AnnotateTests(unittest.TestCase):
    def test_annotate_returns_rgb_copy_with_pixels_drawn(self):
        import numpy as np

        from yolostudio.inference import annotate

        frame = np.zeros((100, 200, 3), dtype=np.uint8)  # black BGR frame
        out = annotate(frame, [Detection("mask", 0.9, 20, 20, 80, 80)], Roi(0.1, 0.1, 0.9, 0.9))
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(int(frame.sum()), 0, "the input frame must not be modified")
        self.assertGreater(int(out.sum()), 0)
        # the box colour (255, 0, 0) is red in RGB, so the red channel must contain it
        self.assertTrue((out[..., 0] == 255).any())

    def test_decode_image_rejects_garbage(self):
        from yolostudio.inference import decode_image

        with self.assertRaises(ValueError):
            decode_image(b"not an image")


if __name__ == "__main__":
    unittest.main()
