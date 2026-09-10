from scripts.build_isign_group_splits import bucket, parse_uid


def test_isign_uid_parsing_keeps_source_video_together_across_known_formats():
    assert parse_uid("1782bea75c7d-10") == ("1782bea75c7d", "10")
    assert parse_uid("0pth6-vW81w_e2") == ("0pth6-vW81w", "e2")
    assert parse_uid("completevideoid") == ("completevideoid", None)
    assert bucket("source-video") == bucket("source-video")
