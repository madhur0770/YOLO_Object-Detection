"""End-to-end checks with real weights. Skipped unless YOLO_REAL_TESTS=1 (CI sets it; it needs network access)."""
import os
import unittest

REAL = os.environ.get("YOLO_REAL_TESTS") == "1"


@unittest.skipUnless(REAL, "set YOLO_REAL_TESTS=1 to run tests that download weights")
class RealModelTests(unittest.TestCase):
    def test_bundled_mask_model_loads_with_expected_classes(self):
        from yolostudio.inference import load_model
        from yolostudio.models import get_model_spec

        model = load_model(get_model_spec("mask").weights)
        self.assertEqual(sorted(model.names.values()), ["mask", "no-mask"])

    def test_coco_model_detects_people_in_a_sample_photo(self):
        import urllib.request

        from yolostudio.inference import decode_image, detect, load_model

        url = "https://ultralytics.com/images/bus.jpg"
        with urllib.request.urlopen(url, timeout=60) as response:
            frame = decode_image(response.read())
        detections = detect(load_model("yolo11n.pt"), frame, confidence=0.25)
        labels = [d.label for d in detections]
        self.assertIn("person", labels)
        self.assertIn("bus", labels)

    def test_tracking_assigns_ids_on_repeated_frames(self):
        import urllib.request

        from yolostudio.inference import decode_image, detect, load_model

        with urllib.request.urlopen("https://ultralytics.com/images/bus.jpg", timeout=60) as response:
            frame = decode_image(response.read())
        model = load_model("yolo11n.pt")
        for _ in range(2):
            detections = detect(model, frame, confidence=0.25, track=True)
        self.assertTrue(any(d.track_id is not None for d in detections))


if __name__ == "__main__":
    unittest.main()
