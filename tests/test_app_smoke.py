import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipIf(importlib.util.find_spec("streamlit") is None, "streamlit is not installed")
class AppSmokeTests(unittest.TestCase):
    def test_empty_state_renders_without_loading_a_model(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "YOLO Object Detection Studio")
        self.assertGreaterEqual(len(app.tabs), 3)
        self.assertTrue(any("Upload an image" in i.value for i in app.info))

    def test_invalid_roi_shows_an_error(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.slider[1].set_value((50, 50)).run()  # horizontal range with x1 == x2
        self.assertFalse(app.exception)
        self.assertTrue(any("x1 < x2" in e.value for e in app.sidebar.error))


if __name__ == "__main__":
    unittest.main()
