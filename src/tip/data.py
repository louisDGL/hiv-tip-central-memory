"""Loader for the base paper's infant-macaque viral-load tables.

Files live in the cloned upstream repository and are NOT vendored here:

    external/HIV-SIV-TIP-Modeling/SIV-TIP-MCMC/data/
        PVLInfant_CTR_10 TCID50.csv      3 control animals
        PVLInfant_TIP_10_TCID50.csv      4 TIP-treated animals

Layout is one row per timepoint, first column "Time (weeks)", one column per
animal ID, blanks for missing observations. Values are plasma viral load in
RNA copies/mL. The assay floor appears in the tables as the literal value 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["VLSeries", "load_viral_loads", "DATA_DIR", "ASSAY_FLOOR"]

DATA_DIR = (Path(__file__).resolve().parents[2] / "external"
            / "HIV-SIV-TIP-Modeling" / "SIV-TIP-MCMC" / "data")

CONTROL_FILE = "PVLInfant_CTR_10 TCID50.csv"
TREATED_FILE = "PVLInfant_TIP_10_TCID50.csv"

#: Values at or below this are pre-inoculation / below assay floor, not data.
ASSAY_FLOOR = 5.0

DAYS_PER_WEEK = 7.0


@dataclass(frozen=True)
class VLSeries:
    """One animal's viral-load trajectory, missing points already dropped."""
    animal: str
    arm: str                 # "control" or "treated"
    t_days: np.ndarray
    vl: np.ndarray           # RNA copies/mL

    def truncate(self, t_max_days: float) -> "VLSeries":
        m = self.t_days <= t_max_days
        return VLSeries(self.animal, self.arm, self.t_days[m], self.vl[m])

    @property
    def t_weeks(self) -> np.ndarray:
        return self.t_days / DAYS_PER_WEEK


def _read_table(path: Path, arm: str) -> list[VLSeries]:
    rows = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    header = rows[0].split(",")
    animals = [h.strip() for h in header[1:]]

    times: list[float] = []
    cols: list[list[float]] = [[] for _ in animals]
    for row in rows[1:]:
        cells = row.split(",")
        try:
            t = float(cells[0])
        except ValueError:
            continue
        times.append(t * DAYS_PER_WEEK)
        for j in range(len(animals)):
            raw = cells[j + 1].strip() if j + 1 < len(cells) else ""
            cols[j].append(float(raw) if raw else np.nan)

    t_arr = np.asarray(times, dtype=float)
    out = []
    for name, col in zip(animals, cols):
        v = np.asarray(col, dtype=float)
        keep = np.isfinite(v) & (v > ASSAY_FLOOR) & (t_arr >= 0.0)
        out.append(VLSeries(name, arm, t_arr[keep], v[keep]))
    return out


def load_viral_loads(data_dir: Path = DATA_DIR) -> dict[str, list[VLSeries]]:
    """Return {"control": [...], "treated": [...]}.

    Raises FileNotFoundError with a usable message if the upstream repo has
    not been cloned, since it is gitignored and absent from a fresh checkout.
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} not found. Clone the upstream repository first:\n"
            "  git clone https://github.com/khetanneha/HIV-SIV-TIP-Modeling.git"
            " external/HIV-SIV-TIP-Modeling"
        )
    return {
        "control": _read_table(data_dir / CONTROL_FILE, "control"),
        "treated": _read_table(data_dir / TREATED_FILE, "treated"),
    }
