"""Shared behavior labels used by training/inference."""

BEHAVIOR_LABELS = [
    "normal_flow",
    "abnormal_gathering",
    "pushing_shoving",
    "fighting",
    "vandalism",
]

BEHAVIOR_TO_INDEX = {label: idx for idx, label in enumerate(BEHAVIOR_LABELS)}
INDEX_TO_BEHAVIOR = {idx: label for label, idx in BEHAVIOR_TO_INDEX.items()}
