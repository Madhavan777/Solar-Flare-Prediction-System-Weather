"""Solar Flare Prediction & Space Weather Alert System.

An academic Project-Based Learning prototype that forecasts whether a solar
active region will produce a major (GOES M- or X-class) flare in the 24 hours
after a 12-hour observation window of SDO/HMI SHARP data.

**This is not an operational space-weather warning service.** It is trained and
evaluated on historical SWAN-SF data covering 2010-05-01 to 2018-08-17, consumes
no live feed, and is recall-oriented by design: on the locked test partition it
detects 91.6 % of major flares while 83.8 % of the alerts it raises are false.

The published results are frozen. This package verifies them, adds labelled
post-hoc analysis, and builds the demo — it never retrains over ``models/`` or
rewrites anything in ``results/``.

Entry points::

    python -m solarflare --help
    python -m solarflare evaluate     # recompute every frozen number and compare
    python -m solarflare all          # the whole verification and build pipeline
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "data",
    "env",
    "features",
    "metrics",
    "models",
    "paths",
    "plotting",
]
