#!/usr/bin/env bash
# List files changed upstream since Aviary's last explicit Colibri merge (a683028).
# Usage: sync_port.sh [hy3|colibri]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
BASE="${1:-colibri}"
case "$BASE" in
  hy3) REF="hy3/main" ;;
  colibri) REF="colibri/main" ;;
  *) echo "usage: $0 [hy3|colibri]" >&2; exit 1 ;;
esac
ANCHOR="a683028"
git fetch "$BASE" main 2>/dev/null || true
echo "Porting candidates: diff $ANCHOR..$REF (excluding Aviary overlay)"
echo ""
git diff --name-only "$ANCHOR".."$REF" 2>/dev/null \
  | grep -v '^c/aviary/' \
  | grep -v '^c/cluster_rpc.h$' \
  | grep -v '^c/cluster_telemetry.h$' \
  | grep -v '^c/tests/test_aviary_' \
  | grep -v '^web/src/Cluster.tsx$' \
  | grep -v 'aviary-cluster-plan' \
  | head -80
echo ""
echo "Review with: git diff $ANCHOR..$REF -- <path>"
echo "Never blind-copy: c/coli c/openai_server.py c/hy3.c c/qwen3_moe.c (re-apply Aviary hooks)"
