import unittest

from yolostudio.hosting import is_hosted


class HostingTests(unittest.TestCase):
    def test_streamlit_cloud_path(self):
        self.assertTrue(is_hosted({}, "/mount/src/yolo_object-detection/yolostudio/hosting.py"))

    def test_local_path(self):
        self.assertFalse(is_hosted({}, "C:/Users/me/YOLO/yolostudio/hosting.py"))

    def test_hugging_face_space(self):
        self.assertTrue(is_hosted({"SPACE_ID": "user/space"}, "/app/yolostudio/hosting.py"))

    def test_override_wins(self):
        self.assertFalse(is_hosted({"YOLO_STUDIO_HOSTED": "0"}, "/mount/src/x/yolostudio/hosting.py"))
        self.assertTrue(is_hosted({"YOLO_STUDIO_HOSTED": "1"}, "/home/me/yolostudio/hosting.py"))


if __name__ == "__main__":
    unittest.main()
