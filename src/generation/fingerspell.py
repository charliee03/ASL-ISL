"""ISL fingerspelling: concatenate alpha_keypoints frames into a word sequence.

For letters with multiple variants (e.g. c1/c2, e1/e2/e3), the first variant
found (sorted alphabetically) is always used.

Usage::

    from src.generation.fingerspell import build_fingerspell_sequence

    data = build_fingerspell_sequence("hello")
    # data == {"fps": 25.0, "frames": [...]}
    # Feed directly to animate.process_file() or process_file(data, ...)
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate the alpha_keypoints directory
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_PROJECT_DIR = _HERE.parents[2]
ALPHA_KEYPOINTS_DIR = _PROJECT_DIR / "data" / "isl" / "alpha_keypoints"

# ---------------------------------------------------------------------------
# Build a letter → first-variant-path mapping at import time
# ---------------------------------------------------------------------------

def _build_letter_map() -> dict[str, Path]:
    """Return {letter: path_to_first_variant_json} for every a–z letter found."""
    if not ALPHA_KEYPOINTS_DIR.is_dir():
        logger.warning("alpha_keypoints directory not found: %s", ALPHA_KEYPOINTS_DIR)
        return {}

    letter_map: dict[str, list[Path]] = {}
    for json_path in sorted(ALPHA_KEYPOINTS_DIR.glob("isl_alpha_*.json")):
        # stem examples: isl_alpha_a, isl_alpha_c1, isl_alpha_e2
        stem = json_path.stem  # e.g. "isl_alpha_c1"
        suffix = stem[len("isl_alpha_"):]  # e.g. "c1", "a", "e2"
        if not suffix:
            continue
        # The letter is the leading alpha character(s) — strip trailing digits
        letter = suffix.rstrip("0123456789")
        if not letter or not letter.isalpha():
            continue
        letter = letter.lower()
        letter_map.setdefault(letter, []).append(json_path)

    # Keep only the first (alphabetically sorted files ensure c1 < c2 etc.)
    return {letter: paths[0] for letter, paths in letter_map.items()}


LETTER_KEYPOINTS: dict[str, Path] = _build_letter_map()

logger.info(
    "Fingerspell: %d letters loaded from %s. Variants available for: %s",
    len(LETTER_KEYPOINTS),
    ALPHA_KEYPOINTS_DIR,
    ", ".join(
        letter
        for letter in sorted(LETTER_KEYPOINTS)
        if any(
            p.stem.rstrip("0123456789")[len("isl_alpha_"):] == letter
            and any(
                q.stem.rstrip("0123456789")[len("isl_alpha_"):] == letter and q != p
                for q in ALPHA_KEYPOINTS_DIR.glob("isl_alpha_*.json")
            )
            for p in [LETTER_KEYPOINTS[letter]]
        )
    ) or "none",
)


# ---------------------------------------------------------------------------
# Frame-loading with a simple in-process cache
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _load_letter_frames(path: Path) -> tuple[float, list]:
    """Load and cache the fps + frames list for one letter keypoint file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    fps = float(data.get("fps", 25.0))
    frames = data.get("frames", [])
    return fps, frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_fingerspell_sequence(word: str, hold_frames: int = 5) -> dict | None:
    """Return a keypoint sequence dict that spells out *word* letter by letter.

    Parameters
    ----------
    word:
        The word to fingerspell (case-insensitive). Non-alphabetic characters
        are silently skipped.
    hold_frames:
        Number of times the last frame of each letter is repeated before the
        next letter begins — creates a brief visual pause (~200 ms at 25 fps).

    Returns
    -------
    dict | None
        ``{"fps": float, "frames": [...]}`` ready for ``animate.process_file``,
        or ``None`` if no letter in *word* has keypoint data.
    """
    word_lower = word.lower().strip()
    if not word_lower:
        return None

    all_frames: list = []
    fps_used: float = 25.0
    letters_found = 0

    for char in word_lower:
        if not char.isalpha():
            # Skip digits, punctuation, spaces silently
            continue

        path = LETTER_KEYPOINTS.get(char)
        if path is None:
            logger.warning("Fingerspell: no keypoint data for letter %r — skipping", char)
            continue

        try:
            fps, frames = _load_letter_frames(path)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Fingerspell: could not load %s: %s", path, exc)
            continue

        if not frames:
            logger.warning("Fingerspell: empty frame list in %s", path)
            continue

        fps_used = fps
        # Tag each frame with the letter label for on-screen display
        label = char.upper()
        labelled = [{**f, "label": label} for f in frames]
        all_frames.extend(labelled)
        letters_found += 1

        # Brief hold — repeat the last frame to separate letters visually
        if hold_frames > 0:
            last_frame = labelled[-1]
            all_frames.extend([last_frame] * hold_frames)

    if not all_frames:
        logger.warning(
            "Fingerspell: could not build any frames for word %r "
            "(no alphabetic characters had keypoint data)",
            word,
        )
        return None

    logger.debug(
        "Fingerspell: built %d frames for %r (%d letters, hold=%d)",
        len(all_frames),
        word,
        letters_found,
        hold_frames,
    )
    return {"fps": fps_used, "frames": all_frames}
