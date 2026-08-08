# Tools

Offline scripts for model preparation and engineering work in the
[Aviary](https://github.com/SensAI-PT/aviary-hy3) / [Colibri](https://github.com/JustVugg/colibri)
tree. They are **not** runtime dependencies of the C engine or the Aviary control plane
(`c/aviary/`).

- `convert_fp8_to_int4.py`, `download_glm52.py`: model preparation
- `repack_fp8_passthrough.py`: fmt=8 repack (byte-preserved FP8, resident kinds only;
  see the module docstring -- synthetic-fixture-tested only, no real-shard runs yet)
- `make_glm_oracle.py`, `make_glm_bench_model.py`: deterministic fixtures
- `benchmark_cuda_fixture.py`, `eval_glm.py`, `fetch_benchmarks.py`: benchmarks
- `gen_unicode.py`: tokenizer table generation

Run them from `c/`, for example:

```sh
python3 tools/convert_fp8_to_int4.py --selftest
python3 tools/make_glm_bench_model.py --output /tmp/aviary-bench
```
