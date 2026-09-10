import numpy as np
import pytest

from scripts.extract_msasl_landmarks import sample_indices
from scripts.build_msasl_splits import build_splits


def test_sample_indices_cover_video_boundaries():
    assert sample_indices(10, 4).tolist() == [0, 3, 6, 9]


def test_sample_indices_reject_empty_video():
    with pytest.raises(ValueError, match="no frames"):
        sample_indices(0, 32)


def test_schema_has_expected_feature_width():
    empty = np.zeros((32, 33 + 21 + 21, 3), dtype=np.float32)
    assert empty.shape == (32, 75, 3)


def test_split_builder_filters_tracking_failures_and_preserves_official_split():
    metadata = {"samples": [
        {"id": "a", "class_id": 0, "gloss": "hello", "split": "train", "signer": "1", "pose_coverage": 1.0, "hand_coverage": 0.5},
        {"id": "b", "class_id": 0, "gloss": "hello", "split": "val", "signer": "2", "pose_coverage": 1.0, "hand_coverage": 0.0},
    ]}
    splits, vocabulary, rejected = build_splits(metadata, 0.8, 0.1)
    assert vocabulary == ["hello"]
    assert [row["id"] for row in splits["train"]] == ["a"]
    assert splits["val"] == []
    assert rejected["low_hand_coverage"] == 1


def test_split_builder_rejects_signer_leakage():
    metadata = {"samples": [
        {"id": "a", "class_id": 0, "gloss": "hello", "split": "train", "signer": "1", "pose_coverage": 1.0, "hand_coverage": 1.0},
        {"id": "b", "class_id": 0, "gloss": "hello", "split": "test", "signer": "1", "pose_coverage": 1.0, "hand_coverage": 1.0},
    ]}
    with pytest.raises(ValueError, match="Signer leakage"):
        build_splits(metadata, 0.8, 0.1)
