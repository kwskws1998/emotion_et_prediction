"""Shared constants for CMCL-style ET prediction."""

from pathlib import Path

FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]

CMCL_TARGET_STATS = {
    "nFix": {"mean": 15.10, "std": 9.42},
    "FFD": {"mean": 3.19, "std": 1.42},
    "GPT": {"mean": 6.35, "std": 5.91},
    "TRT": {"mean": 5.31, "std": 3.64},
    "fixProp": {"mean": 67.06, "std": 26.06},
}

CMCL_NONZERO_TARGET_STATS = {
    "nFix": {"mean": 15.2315, "std": 9.3603},
    "FFD": {"mean": 3.2184, "std": 1.3973},
    "GPT": {"mean": 6.4000, "std": 5.9024},
    "TRT": {"mean": 5.3552, "std": 3.6237},
    "fixProp": {"mean": 67.6332, "std": 25.4164},
}

DEFAULT_IITB_V2_DIR = Path(
    "data/iitb_sentiment_gaze_raw/extracted/v2/"
    "Eye-tracking_and_SA-II_released_dataset"
)
