# Changelog

All notable changes to Finetorch are documented in this file.

## [Unreleased]

## [0.1.0] - 2026-03-11

### Added

- Initial repository scaffold for Finetorch.
- Rust project layout covering CLI, config, data, model, train, and eval modules.
- Example run configuration in `configs/example_run.toml`.
- README guidance for architecture, data flow, and command usage.
- Placeholder Rust-native CLI and library scaffold for LoRA and QLoRA finetuning workflows.
- Dataset preparation pipeline with JSONL normalization, token counting, deterministic splitting, and sharding.
- Config-driven train and eval commands with modular backend boundaries.
- Initial `llama_cpp` backend scaffold and backend-neutral `LlmBackend` trait.
- Local helper scripts for build, test, lint, local CI, and submodule sync.
- `script-helpers` Git submodule under `scripts/script-helpers`.
- `ci-helpers`-based GitHub Actions for CI and auto-tag automation.
- In-depth project documentation under `docs/`.
- Dedicated getting-started and use-case guides under `docs/`.
- Tag-driven release workflow that publishes the matching `CHANGELOG.md` section as the release body.
