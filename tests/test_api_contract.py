import asyncio
import os
from pathlib import Path

from src.api import server


class _Upload:
    """Minimal asynchronous upload fixture without Starlette thread-pool I/O."""

    def __init__(self, contents: bytes, filename: str, content_type: str):
        self._contents = contents
        self.filename = filename
        self.content_type = content_type

    async def read(self, _size: int = -1) -> bytes:
        return self._contents


def test_avatar_endpoint_is_registered_once_and_returns_playable_url():
    avatar_routes = [
        route for route in server.app.routes
        if getattr(route, "path", None) == "/generate-avatar" and "POST" in getattr(route, "methods", set())
    ]
    assert len(avatar_routes) == 1

    response = asyncio.run(server.generate_avatar({"isl_gloss": "NAMASKAR"}))
    assert response["mode"] in {"landmark_playback", "illustrative_2d"}
    assert response["video_url"].startswith("/generated/avatar-")

    video_path = server.GENERATED_DIR / Path(response["video_url"]).name
    assert video_path.is_file()
    video_path.unlink()


def test_web_client_root_route_is_registered():
    root_routes = [
        route for route in server.app.routes
        if getattr(route, "path", None) == "/" and "GET" in getattr(route, "methods", set())
    ]
    assert len(root_routes) == 1
    assert (server.WEB_DIR / "index.html").is_file()


def test_translation_endpoint_returns_draft_gloss_for_manual_text():
    response = asyncio.run(server.translate_text({"asl_gloss": "nice to meet you"}))
    assert response["asl_gloss"] == "nice to meet you"
    assert response["isl_gloss"]
    assert response["translation_mode"] == "draft_rule_based"
    assert response["review_required"] is True


def test_sentence_lookup_normalises_case_whitespace_and_punctuation():
    assert server._normalise_sentence_lookup(" He is going into the room! ") == "he is going into the room"


def test_sentence_lookup_accepts_safe_near_match_without_changing_negation():
    asset, match_type, score = server._find_cslrt_sentence_asset(
        "He is going to the room", allow_approximate=True
    )
    if server.CSLRT_SENTENCE_ASSETS:
        assert asset is not None
        assert asset["sentence"] == "He is going into the room"
        assert match_type == "approximate"
        assert score is not None and score >= 0.88

        negative_asset, _, _ = server._find_cslrt_sentence_asset(
            "He is not going to the room", allow_approximate=True
        )
        assert negative_asset is None or "not" in negative_asset["sentence"].lower()


def test_generated_video_cleanup_removes_only_expired_avatar_outputs(tmp_path, monkeypatch):
    old_avatar = tmp_path / "avatar-old.mp4"
    old_raw = tmp_path / ".avatar-old.raw.mp4"
    recent_avatar = tmp_path / "avatar-recent.mp4"
    unrelated_file = tmp_path / "keep.txt"
    for path in (old_avatar, old_raw, recent_avatar, unrelated_file):
        path.write_bytes(b"test")
    os.utime(old_avatar, (10, 10))
    os.utime(old_raw, (10, 10))
    os.utime(recent_avatar, (99, 99))
    os.utime(unrelated_file, (10, 10))
    monkeypatch.setattr(server, "GENERATED_DIR", tmp_path)
    monkeypatch.setattr(server, "GENERATED_VIDEO_TTL_SECONDS", 50)

    assert server._cleanup_generated_videos(now=100) == 2
    assert not old_avatar.exists()
    assert not old_raw.exists()
    assert recent_avatar.exists()
    assert unrelated_file.exists()


def test_video_endpoint_rejects_unsupported_media_type_before_processing(monkeypatch):
    upload = _Upload(b"not a video", "notes.txt", "text/plain")
    monkeypatch.setattr(server, "RECOGNITION_ENABLED", True)
    monkeypatch.setattr(server, "model", object())

    response = asyncio.run(server.predict_sequence(upload))

    assert response.status_code == 415


def test_video_endpoint_rejects_oversized_upload_before_processing(monkeypatch):
    upload = _Upload(b"12345", "sign.mp4", "video/mp4")
    monkeypatch.setattr(server, "RECOGNITION_ENABLED", True)
    monkeypatch.setattr(server, "model", object())
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 4)

    response = asyncio.run(server.predict_sequence(upload))

    assert response.status_code == 413
