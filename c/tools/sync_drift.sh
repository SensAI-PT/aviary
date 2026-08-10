#!/usr/bin/env bash
# Print ahead/behind commit counts vs Hy3 fork and upstream Colibri.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

fetch() {
  git fetch "$1" main 2>/dev/null || true
}

hy3_ref="hy3/main"
colibri_ref="colibri/main"

git remote get-url hy3 &>/dev/null || git remote add hy3 https://github.com/ErikTromp/colibri-hy3.git
git remote get-url colibri &>/dev/null || git remote add colibri https://github.com/JustVugg/colibri.git

fetch hy3
fetch colibri

count() {
  git rev-list --left-right --count "$1"...HEAD 2>/dev/null || echo "? ?"
}

hy3_counts=$(count "$hy3_ref")
colibri_counts=$(count "$colibri_ref")

printf "Aviary HEAD: %s\n\n" "$(git log -1 --oneline)"
printf "%-14s ahead  behind\n" "remote"
printf "%-14s %s\n" "hy3/main:" "$(echo "$hy3_counts" | awk '{print $2, $1}')"
printf "%-14s %s\n" "colibri/main:" "$(echo "$colibri_counts" | awk '{print $2, $1}')"
printf "\nSee docs/COLIBRI_SYNC.md for merge procedure.\n"
