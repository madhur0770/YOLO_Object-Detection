import unittest
from pathlib import Path

from yolostudio.models import MODELS, MODELS_DIR, get_model_spec


class ModelRegistryTests(unittest.TestCase):
    def test_keys_are_unique(self):
        keys = [m.key for m in MODELS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_lookup(self):
        self.assertEqual(get_model_spec("mask").title.split()[0], "Mask")
        with self.assertRaises(KeyError):
            get_model_spec("nope")

    def test_bundled_mask_weights_exist_and_are_small_enough_for_git(self):
        weights = Path(get_model_spec("mask").weights)
        self.assertTrue(weights.is_file(), weights)
        self.assertLess(weights.stat().st_size, 50 * 1024 * 1024)
        self.assertEqual(weights.parent, MODELS_DIR)


if __name__ == "__main__":
    unittest.main()
