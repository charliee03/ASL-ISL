import json

from src.generation import fingerspell


def _frame(seed: float) -> dict:
    """Return the minimum valid landmark frame shape for sequencing tests."""
    point = [seed, seed, 0.0]
    return {
        "pose": [point] * 33,
        "left_hand": [point] * 21,
        "right_hand": [point] * 21,
    }


def test_fingerspell_sequences_letters_labels_and_holds(tmp_path, monkeypatch):
    n_path = tmp_path / "isl_alpha_n.json"
    a_path = tmp_path / "isl_alpha_a.json"
    n_path.write_text(json.dumps({"fps": 20, "frames": [_frame(1.0)]}), encoding="utf-8")
    a_path.write_text(json.dumps({"fps": 20, "frames": [_frame(2.0)]}), encoding="utf-8")
    monkeypatch.setattr(fingerspell, "LETTER_KEYPOINTS", {"n": n_path, "a": a_path})
    fingerspell._load_letter_frames.cache_clear()

    sequence = fingerspell.build_fingerspell_sequence("N-a9", hold_frames=1)

    assert sequence is not None
    assert sequence["fps"] == 20.0
    assert [frame["label"] for frame in sequence["frames"]] == ["N", "N", "A", "A"]
    assert sequence["frames"][0]["pose"] == _frame(1.0)["pose"]
    assert sequence["frames"][2]["pose"] == _frame(2.0)["pose"]


def test_fingerspell_returns_none_when_no_letter_asset_exists(monkeypatch):
    monkeypatch.setattr(fingerspell, "LETTER_KEYPOINTS", {})

    assert fingerspell.build_fingerspell_sequence("Nandita") is None
