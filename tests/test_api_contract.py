import asyncio
from pathlib import Path

from src.api import server


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
