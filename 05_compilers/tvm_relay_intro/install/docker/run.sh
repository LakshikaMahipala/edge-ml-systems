#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker build -t tvm-relay-intro -f "${ROOT}/install/docker/Dockerfile" "${ROOT}/install/docker"
docker run --rm -it -v "${ROOT}:/work" tvm-relay-intro bash
