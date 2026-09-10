import json
from pathlib import Path

from scripts.build_cslrt_splits import build_splits
from scripts.extract_cslrt_landmarks import normalize_frame
from src.recognition.dataset import CSLRTDataset, RandomKeypointMask, RandomLandmarkNoise


def _sequence(path: Path, hands: bool = True) -> None:
    hand = [[1.0, 0.0, 0.0]] if hands else [[0.0, 0.0, 0.0]]
    path.write_text(json.dumps({"frames": [{"left_hand": hand, "right_hand": hand}]}))


def test_build_splits_is_signer_independent(tmp_path):
    metadata = []
    for signer in range(1, 8):
        filename = f"sample_{signer}.json"
        _sequence(tmp_path / filename)
        metadata.append({"sentence": "hello", "signer": str(signer), "landmarks": filename, "num_frames": 1, "pose_coverage": 1.0})

    splits, vocabulary = build_splits(metadata, tmp_path)

    assert vocabulary == ["hello"]
    assert {row["signer"] for row in splits["train"]} == {"1", "2", "3", "4", "5"}
    assert {row["signer"] for row in splits["val"]} == {"6"}
    assert {row["signer"] for row in splits["test"]} == {"7"}
    assert len({row["id"] for rows in splits.values() for row in rows}) == 7


def test_cslrt_dataset_resamples_all_75_landmarks(tmp_path):
    keypoints = tmp_path / "keypoints"
    keypoints.mkdir()
    frame = {
        "pose": [[1.0, 2.0, 3.0]] * 33,
        "left_hand": [[4.0, 5.0, 6.0]] * 21,
        "right_hand": [[7.0, 8.0, 9.0]] * 21,
    }
    (keypoints / "sample.json").write_text(json.dumps({"feature_schema_version": "2.0", "frames": [frame, frame]}))
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps([{"landmarks": "sample.json", "class_id": 0, "sentence": "hello"}]))
    vocabulary = tmp_path / "vocabulary.json"
    vocabulary.write_text(json.dumps(["hello"]))

    dataset = CSLRTDataset(keypoints, manifest, vocabulary, num_frames=4)
    sample = dataset[0]

    assert tuple(sample["keypoints"].shape) == (4, 75, 3)
    assert sample["label"] == 0
    assert sample["gloss"] == "hello"


def test_landmark_noise_preserves_missing_points():
    import numpy as np

    keypoints = np.zeros((2, 3, 3), dtype=np.float32)
    keypoints[0, 0] = [1.0, 1.0, 1.0]
    sample = RandomLandmarkNoise(std=0.1)({"keypoints": keypoints})
    assert np.array_equal(sample["keypoints"][1], np.zeros((3, 3)))
    assert not np.array_equal(sample["keypoints"][0, 0], keypoints[0, 0])


def test_normalization_keeps_missing_hands_zero():
    import numpy as np

    pose = np.zeros((33, 3), dtype=np.float32)
    pose[11] = [0.25, 0.5, 0.0]
    pose[12] = [0.75, 0.5, 0.0]
    hand = np.zeros((21, 3), dtype=np.float32)
    normalized_pose, left, right, tracked = normalize_frame(pose, hand, hand)
    assert tracked
    assert np.all(left == 0) and np.all(right == 0)
    assert np.allclose((normalized_pose[11] + normalized_pose[12]) / 2, 0)
    assert np.isclose(np.linalg.norm(normalized_pose[11, :2] - normalized_pose[12, :2]), 1)


def test_keypoint_mask_masks_complete_xyz_triplets():
    import numpy as np

    keypoints = np.ones((2, 3, 3), dtype=np.float32)
    sample = RandomKeypointMask(probability=1.0)({"keypoints": keypoints})
    assert np.count_nonzero(sample["keypoints"]) == 0
