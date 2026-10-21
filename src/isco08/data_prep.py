from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_labels(isco_dir: Path) -> pd.DataFrame:
    path = Path(isco_dir) / "labels.csv"
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)
    labels["code"] = labels["code"].str.zfill(4)
    return labels
