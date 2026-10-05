"""Build the JavaScript/Python parity fixture.

Extracts a deterministic sample of real SWAN-SF windows into
``tests/fixtures/parity_windows.json.gz`` so that ``tests/test_js_parity.py``
can compare the browser feature extractor against the Python one on genuine
data, on any machine, without the 5.7 GB of raw tarballs.

Partition 4 is used because it is an intact archive on this machine (P2 and P5
are corrupt gzip streams — see docs/REPRODUCIBILITY.md) and because feature
extraction is partition-independent: the parity question is purely about
arithmetic, not about which split a window belongs to.

Usage:
    python scripts/build_parity_fixture.py [n_windows] [partition]
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import SHARP
from solarflare import paths

N_WINDOWS = int(sys.argv[1]) if len(sys.argv) > 1 else 220
PARTITION = int(sys.argv[2]) if len(sys.argv) > 2 else 4
STRIDE = 97  # a prime stride, so the sample spans the archive rather than its head
OUT = paths.ROOT / "tests" / "fixtures" / "parity_windows.json.gz"


def main() -> int:
    tarball = paths.RAW_DATA / f"partition{PARTITION}_instances.tar.gz"
    if not tarball.is_file():
        print(f"FAIL - {paths.relative(tarball)} not found.")
        return 1

    print(
        f"streaming {paths.relative(tarball)}, taking every {STRIDE}th window "
        f"until {N_WINDOWS} are collected..."
    )
    windows: list[dict] = []
    seen = 0
    with tarfile.open(tarball, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            seen += 1
            if seen % STRIDE:
                continue
            raw = archive.extractfile(member).read()
            frame = pd.read_csv(io.BytesIO(raw), sep="\t", usecols=["Timestamp", *SHARP])
            windows.append(
                {
                    "file": member.name.split("/")[-1],
                    "folder": member.name.split("/")[-2],
                    "n_rows": len(frame),
                    "columns": {
                        parameter: [
                            None if not np.isfinite(v) else float(v)
                            for v in frame[parameter].to_numpy(dtype=np.float64)
                        ]
                        for parameter in SHARP
                    },
                }
            )
            if len(windows) % 25 == 0:
                print(f"  {len(windows)}/{N_WINDOWS}", flush=True)
            if len(windows) >= N_WINDOWS:
                break

    missing_any = sum(
        1 for w in windows if any(v is None for col in w["columns"].values() for v in col)
    )
    all_nan_columns = sum(
        1 for w in windows for col in w["columns"].values() if all(v is None for v in col)
    )
    print(
        f"collected {len(windows)} windows; {missing_any} contain at least one missing "
        f"value and {all_nan_columns} parameter columns are entirely missing"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": f"raw_data/partition{PARTITION}_instances.tar.gz",
        "stride": STRIDE,
        "n_windows": len(windows),
        "sharp_parameters": list(SHARP),
        "note": "Real SWAN-SF windows, used only to check that the JavaScript feature "
        "extractor agrees with the Python one. Feature extraction does not "
        "depend on which partition a window came from.",
        "windows": windows,
    }
    with gzip.open(OUT, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"wrote {paths.relative(OUT)} ({OUT.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
