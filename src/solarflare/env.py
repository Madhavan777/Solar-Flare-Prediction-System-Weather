"""Runtime environment checks.

The seven pipelines in ``models/`` are scikit-learn pickles. Loading them under a
different scikit-learn minor version either raises, or — worse — succeeds while
silently reconstructing estimators whose internals have changed, so the
published metrics would no longer be reproducible.

:func:`check` therefore fails hard on a scikit-learn mismatch and warns on
everything else.
"""

from __future__ import annotations

import sys
import warnings
from importlib.metadata import PackageNotFoundError, version

#: Versions the published results were produced and verified with.
#: scikit-learn is the only hard requirement; see docs/REPRODUCIBILITY.md.
REQUIRED = {"scikit-learn": "1.8.0"}
EXPECTED = {
    "numpy": "2.4.4",
    "pandas": "3.0.2",
    "scipy": "1.18.1",
    "joblib": "1.5.3",
    "matplotlib": "3.10.9",
}
#: Python 3.11 produced the original results; 3.12 is verified to reproduce them
#: bit-for-bit (docs/DECISIONS.md D-02).
PYTHON_VERIFIED = ("3.11", "3.12")


class EnvironmentMismatch(RuntimeError):
    """Raised when an installed version makes the saved models unusable."""


def installed(package: str) -> str | None:
    """Return the installed version of ``package``, or ``None`` if absent."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def report() -> dict[str, str | None]:
    """Return a mapping of every tracked package to its installed version."""
    out: dict[str, str | None] = {"python": ".".join(map(str, sys.version_info[:3]))}
    for package in (*REQUIRED, *EXPECTED):
        out[package] = installed(package)
    return out


def check(strict: bool = False) -> None:
    """Validate the runtime environment.

    Args:
        strict: if true, any deviation from :data:`EXPECTED` or
            :data:`PYTHON_VERIFIED` is an error rather than a warning.

    Raises:
        EnvironmentMismatch: if scikit-learn is missing or is not the required
            version, or if ``strict`` is set and any other version deviates.
    """
    problems: list[str] = []
    cautions: list[str] = []

    for package, want in REQUIRED.items():
        have = installed(package)
        if have is None:
            problems.append(
                f"{package} is not installed. The saved pipelines cannot be loaded without "
                f"{package}=={want}. Run:  pip install -r requirements.txt"
            )
        elif have != want:
            problems.append(
                f"{package}=={have} is installed but the saved pipelines were built with "
                f"{package}=={want}. Loading them under a different minor version will not "
                f"reproduce the published metrics. Run:  pip install {package}=={want}"
            )

    for package, want in EXPECTED.items():
        have = installed(package)
        if have is None:
            cautions.append(f"{package} is not installed (expected {want})")
        elif have != want:
            cautions.append(f"{package}=={have} installed, results were produced with {want}")

    python = ".".join(map(str, sys.version_info[:2]))
    if python not in PYTHON_VERIFIED:
        cautions.append(
            f"running Python {python}; the results were produced on 3.11 and re-verified on "
            f"3.12. Other versions are untested."
        )

    if problems:
        raise EnvironmentMismatch("\n".join(problems))
    if cautions and strict:
        raise EnvironmentMismatch("\n".join(cautions))
    for caution in cautions:
        warnings.warn(f"environment: {caution}", RuntimeWarning, stacklevel=2)


def format_report() -> str:
    """Return a human-readable, aligned version table."""
    rows = report()
    width = max(len(k) for k in rows)
    lines = []
    for package, have in rows.items():
        want = REQUIRED.get(package) or EXPECTED.get(package)
        if package == "python":
            want = " / ".join(PYTHON_VERIFIED)
        flag = "" if have and want and (have == want or package == "python") else "  <-- differs"
        lines.append(f"  {package:<{width}}  {have or 'MISSING':<10}  (expected {want}){flag}")
    return "\n".join(lines)
