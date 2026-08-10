# Colibri upstream sync

Aviary (`SensAI-PT/aviary-hy3`) is a **third fork** in a three-way relationship:

| repository | role |
|---|---|
| [SensAI-PT/aviary-hy3](https://github.com/SensAI-PT/aviary-hy3) | Aviary cluster product (this repo) |
| [ErikTromp/colibri-hy3](https://github.com/ErikTromp/colibri-hy3) | Hy3 / Qwen3 engine fork base |
| [JustVugg/colibri](https://github.com/JustVugg/colibri) | Upstream Colibri engines, UI, CI |

“Behind upstream” does **not** mean broken — Aviary adds ~1200 cluster/Hy3 commits while upstream
moved independently. Run `c/tools/sync_drift.sh` to see current ahead/behind counts.

## Remotes

```bash
git remote add origin git@github.com:SensAI-PT/aviary-hy3.git
git remote add hy3 https://github.com/ErikTromp/colibri-hy3.git
git remote add colibri https://github.com/JustVugg/colibri.git
git fetch hy3 main
git fetch colibri main
```

## Selective merge procedure

**Note:** After the repository history rewrite (`git filter-repo`, fc66d4b), Aviary no longer
shares a Git merge-base with `hy3/main` or `colibri/main`. A plain `git merge` will fail with
*refusing to merge unrelated histories*. Use **diff-based porting** instead:

```bash
git checkout -b sync/upstream-$(date +%Y-%m)
git tag aviary-pre-sync

# See what upstream changed since our last explicit merge (Colibri v1.5.0)
git fetch hy3 main && git fetch colibri main
./c/tools/sync_port.sh colibri   # list porting candidates
./c/tools/sync_port.sh hy3

# Port one file at a time — never overwrite Aviary overlay paths (see preserve list)
git diff a683028..colibri/main -- c/colibri.c | less
# manually apply useful hunks, re-apply Aviary hooks in coli/openai_server/hy3/qwen3_moe
```

When histories share a merge-base again (fresh clone + merge without rewrite), the classic
merge flow applies:

```bash
git merge hy3/main -m "Merge hy3/main into Aviary sync branch"
git merge colibri/main -m "Merge colibri/main into Aviary sync branch"
```

### Always keep (Aviary overlay — prefer ours)

- Entire `c/aviary/` directory
- `c/cluster_rpc.h`, `c/cluster_telemetry.h`
- Aviary tests: `c/tests/test_aviary_*.py`, `c/tests/test_cluster_*.py`, `c/tools/cluster_bench.py`
- `web/src/Cluster.tsx` and cluster i18n strings
- `docs/AVIARY.md`, `docs/cluster_protocol.md`, this file

### Three-way merge manually (preserve Aviary hooks)

| file | keep |
|---|---|
| `c/coli` | `master` / `agent` subcommands |
| `c/hy3.c`, `c/qwen3_moe.c` | `#include cluster_rpc.h`, `cluster_rpc_expert()` in `moe()`, `CLUSTER_*` mux |
| `c/openai_server.py` | `/cluster/*`, `exec_cluster_cmd()`, TRACE parsing, `set_cluster_job()` |
| `web/src/App.tsx`, `web/src/lib/api.ts` | Cluster tab wiring |

### Take from upstream when safe

- Core engines: `colibri.c`, `deepseek_v4`, `kimi_k3`, CUDA backends, CI workflows
- Standard docs and non-cluster web UI improvements
- Hy3/Qwen3 engine fixes **only if** cluster hooks survive the merge

### Reject upstream changes that

- Remove `AVIARY_*` env handling, `coli master`/`agent`, or cluster RPC paths
- Replace Hy3/Qwen3 engines with versions incompatible with `cluster_rpc.h`

## Drift tracking

```bash
./c/tools/sync_drift.sh
```

Example output:

```
hy3/main:     ahead 1235  behind 1040
colibri/main: ahead 1235  behind 1194
```

Last explicit upstream merge in Aviary history: **Colibri v1.5.0** (`a683028`).

## Validation checklist

- [ ] `make qwen3_moe` (or full `./setup.sh`) succeeds
- [ ] `python -m pytest tests/test_aviary_*.py -q` passes
- [ ] `coli master` and `coli agent` subcommands present
- [ ] `AVIARY_CLUSTER=1` agents register and Cluster UI loads
- [ ] Optional: `COLI_MODEL=… pytest tests/test_cluster_oracle.py` (token-exact gate)
