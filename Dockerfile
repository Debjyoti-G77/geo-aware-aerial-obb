# syntax=docker/dockerfile:1
# Pinned CPU base; keep the conda-forge native GIS stack together on Linux.
FROM mambaorg/micromamba:2.9.0@sha256:e0a99b0f17a759e14c2f967dc0ca2d3a3c1ca3c62955f4d20bba770eaaf0184d AS dependencies

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app && chown "$MAMBA_USER:$MAMBA_USER" /app
USER $MAMBA_USER
WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER deploy/environment.container.yml deploy/requirements.container.txt /tmp/dependencies/
# --all keeps in-use extracted caches; remove them only without environment symlinks into them.
RUN micromamba install --yes --name base --file /tmp/dependencies/environment.container.yml \
    && micromamba run --name base python -c "from pathlib import Path; p=Path('/opt/conda'); cache=p/'pkgs'; assert not any(f.is_symlink() and f.resolve().is_relative_to(cache) for f in p.rglob('*') if not f.is_relative_to(cache)), 'Environment symlinks depend on package cache'" \
    && micromamba clean --all --yes \
    && micromamba clean --force-pkgs-dirs --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN python -m pip install --no-cache-dir --only-binary=:all: -r /tmp/dependencies/requirements.container.txt \
    && python -m pip check

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    GEO_OBB_DEVICE=cpu \
    PORT=8000

# Explicit copies are a second boundary beyond .dockerignore.
COPY --chown=$MAMBA_USER:$MAMBA_USER src/ ./src/
COPY --chown=$MAMBA_USER:$MAMBA_USER configs/ ./configs/
COPY --chown=$MAMBA_USER:$MAMBA_USER deploy/ ./deploy/
COPY --chown=$MAMBA_USER:$MAMBA_USER LICENSE THIRD_PARTY_NOTICES.md README.md environment.yml requirements.txt ./
COPY --chown=$MAMBA_USER:$MAMBA_USER LICENSES/ ./LICENSES/
COPY --chown=$MAMBA_USER:$MAMBA_USER docs/ ./docs/

# CI reuses exactly the runtime dependency layer, without model/data assets.
FROM dependencies AS test
COPY --chown=$MAMBA_USER:$MAMBA_USER tests/ ./tests/
CMD ["python", "-B", "-m", "unittest", "discover", "-s", "tests", "-v"]

FROM dependencies AS runtime
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/opt/conda/bin/python", "/app/deploy/healthcheck.py"]
# Preserve micromamba's inherited activation entrypoint. Uvicorn runs one worker.
CMD ["python", "-m", "deploy.serve"]
