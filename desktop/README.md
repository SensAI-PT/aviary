# Aviary desktop

Tauri v2 shell for the shared React dashboard in `../web`.

This directory intentionally contains no second frontend. During development,
Tauri starts the Vite server from `web/`; release builds package `web/dist`.

The desktop app connects to an OpenAI-compatible server configured in the UI —
either a single-node [`coli serve`](https://github.com/JustVugg/colibri) or an
**Aviary master** (`http://host:9000/v1`) for cluster chat and the Cluster tab.

Bundling the inference engine or managing its process is intentionally deferred:
the model is hundreds of gigabytes and must remain an external, user-selected
resource rather than an opaque application sidecar.

## Development

From the repository root:

```sh
cd web
npm ci
cd ../desktop
cargo install tauri-cli --version "^2.0.0" --locked
cargo tauri dev
```

This first desktop increment only packages the existing UI in a native window.
It does not start the inference engine, download models, or add native filesystem
and process permissions.

## Validation

```sh
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo check --manifest-path src-tauri/Cargo.toml
```
