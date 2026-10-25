import unittest
from collections import Counter

from yolostudio.counting import Aggregator, Detection, Roi, build_csv, build_video_csv, count_in_roi, count_labels


def det(label, x1, y1, x2, y2, track=None, conf=0.9):
    return Detection(label, conf, x1, y1, x2, y2, track)


class RoiTests(unittest.TestCase):
    def test_pixels_scale_with_frame_size(self):
        roi = Roi(0.1, 0.2, 0.9, 0.8)
        self.assertEqual(roi.to_pixels(1000, 500), (100, 100, 900, 400))
        self.assertEqual(roi.to_pixels(200, 100), (20, 20, 180, 80))

    def test_contains_uses_the_centre_strictly(self):
        roi = Roi(0.25, 0.25, 0.75, 0.75)  # on a 100x100 frame: 25..75
        self.assertTrue(roi.contains(det("a", 40, 40, 60, 60), 100, 100))
        self.assertFalse(roi.contains(det("a", 0, 0, 20, 20), 100, 100))
        self.assertFalse(roi.contains(det("a", 0, 40, 50, 60), 100, 100))  # centre x == 25 is on the border

    def test_box_larger_than_roi_counts_when_its_centre_is_inside(self):
        self.assertTrue(Roi(0.4, 0.4, 0.6, 0.6).contains(det("a", 0, 0, 100, 100), 100, 100))

    def test_invalid_rois_are_rejected(self):
        for args in [(0.5, 0.1, 0.5, 0.9), (0.6, 0.1, 0.4, 0.9), (0.1, 0.9, 0.9, 0.1), (-0.1, 0, 1, 1), (0, 0, 1.2, 1)]:
            with self.assertRaises(ValueError):
                Roi(*args)


class CountTests(unittest.TestCase):
    def setUp(self):
        self.dets = [det("person", 40, 40, 60, 60), det("person", 0, 0, 10, 10), det("car", 45, 45, 55, 55)]
        self.roi = Roi(0.25, 0.25, 0.75, 0.75)

    def test_count_labels(self):
        self.assertEqual(count_labels(self.dets), Counter({"person": 2, "car": 1}))
        self.assertEqual(count_labels([]), Counter())

    def test_count_in_roi(self):
        self.assertEqual(count_in_roi(self.dets, self.roi, 100, 100), Counter({"person": 1, "car": 1}))

    def test_build_csv(self):
        text = build_csv(Counter({"person": 2, "car": 1}), Counter({"person": 1}))
        self.assertEqual(text.splitlines(), ["class,count,count_in_roi", "car,1,0", "person,2,1", "TOTAL,3,1"])

    def test_build_csv_empty(self):
        self.assertEqual(build_csv(Counter(), Counter()).splitlines(), ["class,count,count_in_roi", "TOTAL,0,0"])


class AggregatorTests(unittest.TestCase):
    def setUp(self):
        self.roi = Roi(0.25, 0.25, 0.75, 0.75)

    def test_peak_is_a_maximum_not_a_sum(self):
        # Regression: the original script summed per-frame counts and reported the result as a maximum.
        agg = Aggregator()
        for _ in range(10):  # the same two people visible for ten frames
            agg.add_frame([det("person", 40, 40, 60, 60), det("person", 41, 41, 61, 61)], self.roi, 100, 100)
        self.assertEqual(agg.frames, 10)
        self.assertEqual(agg.peak["person"], 2)
        self.assertEqual(agg.peak_roi["person"], 2)
        self.assertEqual(agg.detections["person"], 20)

    def test_peak_tracks_the_busiest_frame(self):
        agg = Aggregator()
        agg.add_frame([det("car", 40, 40, 60, 60)], self.roi, 100, 100)
        agg.add_frame([det("car", 40, 40, 60, 60)] * 3, self.roi, 100, 100)
        agg.add_frame([], self.roi, 100, 100)
        self.assertEqual(agg.peak["car"], 3)
        self.assertEqual(agg.frames, 3)

    def test_unique_tracks(self):
        agg = Aggregator()
        agg.add_frame([det("person", 40, 40, 60, 60, track=1), det("person", 41, 41, 61, 61, track=2)], self.roi, 100, 100)
        agg.add_frame([det("person", 40, 40, 60, 60, track=1), det("person", 41, 41, 61, 61, track=3)], self.roi, 100, 100)
        self.assertEqual(len(agg.unique_tracks["person"]), 3)

    def test_video_csv(self):
        agg = Aggregator()
        agg.add_frame([det("person", 40, 40, 60, 60, track=7)], self.roi, 100, 100)
        lines = build_video_csv(agg).splitlines()
        self.assertEqual(lines[0], "class,peak_per_frame,peak_per_frame_in_roi,total_detections,unique_tracks")
        self.assertEqual(lines[1], "person,1,1,1,1")
        self.assertEqual(lines[-1], "FRAMES,1,,,")

    def test_video_csv_leaves_tracks_blank_without_tracking(self):
        agg = Aggregator()
        agg.add_frame([det("person", 40, 40, 60, 60)], self.roi, 100, 100)
        self.assertEqual(build_video_csv(agg).splitlines()[1], "person,1,1,1,")


if __name__ == "__main__":
    unittest.main()
