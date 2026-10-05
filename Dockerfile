# Reproducible environment for the Solar Flare Prediction & Space Weather Alert
# System. Pinned to the exact versions the published results were produced with;
# scikit-learn 1.8.0 in particular is load-bearing, because the seven pipelines
# in models/ are scikit-learn pickles.
#
# Build:
#   docker build -t solarflare .
#
# Verify the published results reproduce (needs the data mounted):
#   docker run --rm -v "$PWD/data:/app/data:ro" solarflare evaluate
#
# Run only what needs no dataset (the models are in the image):
#   docker run --rm solarflare test --fast
#
# Serve the dashboard:
#   docker run --rm -p 8791:8791 solarflare dashboard --port 8791
#   # then open http://localhost:8791/index.html
#
# The image deliberately does NOT contain data/ or raw_data/ — 5.9 GB that is
# not ours to redistribute. See docs/DATA_CARD.md.

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="Solar Flare Prediction & Space Weather Alert System"
LABEL org.opencontainers.image.description="Academic PBL prototype. Not an operational \
space-weather warning service."
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/Madhavan777/Solar-Flare-Prediction-System-Weather"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SOLARFLARE_ROOT=/app

WORKDIR /app

# Dependencies first, so the layer caches independently of the source.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

# The package and everything needed to verify it. data/ and raw_data/ are
# excluded by .dockerignore.
COPY pyproject.toml README.md LICENSE CITATION.cff ./
COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/
COPY figures/ ./figures/
COPY dashboard/ ./dashboard/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

RUN python -m pip install -e . --no-deps

# Fail the build if the environment cannot load the frozen pipelines.
RUN python -m solarflare env

# Fail the build if the published numbers no longer agree with the documents.
RUN python scripts/check_consistency.py

RUN useradd --create-home --uid 10001 solarflare \
 && chown -R solarflare:solarflare /app
USER solarflare

EXPOSE 8791

ENTRYPOINT ["python", "-m", "solarflare"]
CMD ["--help"]
