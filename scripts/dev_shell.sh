#!/usr/bin/env bash
#
# Drop into an interactive shell inside the koral-placer Docker image with the
# repo bind-mounted at /challenge. DREAMPlace is pre-built; placer.py edits
# show up immediately via the mount.
#
# Usage:
#   ./scripts/dev_shell.sh                   # interactive bash
#   ./scripts/dev_shell.sh evaluate -b ibm01 # run a command directly

set -euo pipefail

IMAGE="koral-placer"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not built. Run:" >&2
    echo "  docker build -t $IMAGE -f submissions/koral/Dockerfile ." >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    # Interactive shell
    docker run --rm -it \
        --runtime=nvidia --gpus all \
        --memory 32g \
        -v "$REPO_ROOT":/challenge \
        --entrypoint bash \
        "$IMAGE"
else
    # Run command (e.g. evaluate -b ibm01)
    docker run --rm \
        --runtime=nvidia --gpus all \
        --memory 32g \
        -v "$REPO_ROOT":/challenge \
        --entrypoint "" \
        "$IMAGE" \
        bash -lc "cd /challenge && python -m macro_place.evaluate $*"
fi
