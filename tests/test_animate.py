from src.generation.animate import (
    CANVAS_H,
    CANVAS_W,
    VIEW_MARGIN_BOTTOM,
    VIEW_MARGIN_TOP,
    fitted_view,
    fitted_views_by_token,
    project,
)


def _frame(points):
    return {"pose": points, "left_hand": [], "right_hand": []}


def test_fitted_view_keeps_full_sequence_inside_the_rendering_area():
    frames = [_frame([[-2.0, -1.0, 0.0], [3.0, 5.0, 0.0]])]

    view = fitted_view(frames)
    projected = [project(point, view) for point in frames[0]["pose"]]

    assert all(0 <= x < CANVAS_W for x, _ in projected)
    assert all(VIEW_MARGIN_TOP <= y <= CANVAS_H - VIEW_MARGIN_BOTTOM for _, y in projected)


def test_fitted_view_ignores_word_gap_frames():
    frames = [_frame([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]), {"is_word_gap": True}]

    assert fitted_view(frames) == fitted_view(frames[:1])


def test_composite_words_get_independent_stable_views():
    frames = [
        {**_frame([[0.0, 0.0, 0.0], [0.1, 0.1, 0.0]]), "token_label": "small"},
        {**_frame([[-4.0, -3.0, 0.0], [4.0, 3.0, 0.0]]), "token_label": "large"},
    ]

    views = fitted_views_by_token(frames)

    assert set(views) == {"small", "large"}
    assert views["small"][0] > views["large"][0]
