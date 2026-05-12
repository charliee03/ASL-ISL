import numpy as np
import pytest

from src.recognition.preprocess import HandKeypointExtractor


class TestHandKeypointExtractor:
    def setup_method(self):
        self.extractor = HandKeypointExtractor()

    def test_extract_empty_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp = self.extractor.extract(frame)
        assert isinstance(kp, list)
        assert len(kp) == 0

    def test_extract_rgb_frame(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        kp = self.extractor.extract(frame)
        assert isinstance(kp, list)

    def test_multiple_calls_reuse(self):
        extractor = HandKeypointExtractor()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kp1 = extractor.extract(frame)
        kp2 = extractor.extract(frame)
        assert isinstance(kp1, list)
        assert isinstance(kp2, list)
