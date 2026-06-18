# Finetorch

Finetorch is a Rust-native CLI and library toolkit for practical LLM finetuning on a single GPU. It is designed around lightweight adapter training rather than full pretraining, with clear boundaries between dataset preparation, backend integration, training orchestration, and evaluation.

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration Guide](docs/configuration.md)
- [Getting Started](docs/getting-started.md)
- [CLI Workflows](docs/cli-workflows.md)
- [Use Cases](docs/use-cases.md)
- [Backend and Adapter Design](docs/backends.md)
- [Changelog](CHANGELOG.md)

## Quick Start

Create a small JSONL dataset:

```bash
mkdir -p data
cat > data/train.jsonl <<'EOF'
{"instruction":"Answer briefly","input":"What is LoRA?","output":"LoRA is a parameter-efficient finetuning method."}
{"prompt":"Complete: Gemma is","completion":"a family of language models."}
EOF
```

Prepare shards:

```bash
cargo run -- prepare-dataset \
  --input data/train.jsonl \
  --output artifacts/dataset
```

Run the scaffolded training flow:

```bash
cargo run -- train --config configs/example_run.toml
```

Evaluate a held-out file:

```bash
cargo run -- eval \
  --config configs/example_run.toml \
  --dataset data/train.jsonl
```

## Architecture Overview

Finetorch is split into four primary layers:

1. CLI layer (`src/cli/`)
   - Parses commands and config paths.
   - Orchestrates dataset preparation, training runs, and evaluation jobs.
   - Emits user-facing summaries and output locations.

2. Data layer (`src/data/`)
   - Reads JSONL instruction-tuning data.
   - Normalizes mixed schemas into one internal example format.
   - Applies tokenizer selection and tokenization.
   - Produces shard manifests and train/val split directories.

3. Model layer (`src/model/`)
   - Defines the `LlmBackend` trait for backend-neutral finetuning.
   - Hosts LoRA and QLoRA configuration structs.
   - Wraps backend-specific loading and adapter persistence.
   - Starts with a `llama_cpp` bridge and leaves room for more backends.

4. Training and evaluation layer (`src/train/`, `src/eval/`)
   - Loads config-driven training jobs.
   - Builds optimizer and scheduler state.
   - Runs a lightweight training loop suitable for LoRA/QLoRA adapters.
   - Computes task metrics for small-scale evaluation.

## Data Flow

1. `finetorch prepare-dataset --input data.jsonl --output dataset/`
   - Read JSONL examples.
   - Normalize records into `{ prompt, completion }` pairs.
   - Tokenize with the selected tokenizer.
   - Shuffle and shard into `train/` and `val/` outputs.
   - Write a dataset manifest for downstream runs.

2. `finetorch train --config configs/example_run.toml`
   - Load `run.toml`.
   - Instantiate the selected backend.
   - Load the base model and apply LoRA/QLoRA settings.
   - Run the training loop with optimizer, scheduler, and accumulation settings.
   - Save adapter weights and JSONL training logs.

3. `finetorch eval --config configs/example_run.toml --dataset eval.jsonl`
   - Load config and backend.
   - Read evaluation examples.
   - Run forward passes over the dataset.
   - Compute perplexity, exact match, BLEU, and ROUGE-L summaries.

## Project Structure

```text
src/
  main.rs
  lib.rs
  config.rs
  cli/
    mod.rs
    prepare.rs
    train.rs
    eval.rs
  data/
    mod.rs
    jsonl.rs
    tokenizer.rs
    sharding.rs
  model/
    mod.rs
    backend.rs
    llama_cpp.rs
    lora.rs
  train/
    mod.rs
    loop.rs
    optimizer.rs
    scheduler.rs
  eval/
    mod.rs
    metrics.rs
configs/
  example_run.toml
docs/
  architecture.md
  configuration.md
  getting-started.md
  cli-workflows.md
  use-cases.md
  backends.md
```

## Example Commands

Prepare a dataset:

```bash
cargo run -- prepare-dataset \
  --input data/alpaca_like.jsonl \
  --output artifacts/dataset \
  --tokenizer sentencepiece:models/llama-3/tokenizer.model \
  --train-ratio 0.95 \
  --shard-size 2048
```

Run a small finetuning job:

```bash
cargo run -- train --config configs/example_run.toml
```

Evaluate the resulting adapter:

```bash
cargo run -- eval \
  --config configs/example_run.toml \
  --dataset data/eval.jsonl
```

## Current Scope

This scaffold focuses on:
- LoRA and QLoRA adapter workflows
- Config-driven orchestration
- Dataset preparation and sharding
- Backend extensibility

This scaffold does not yet implement a production-grade GPU training kernel. It establishes the module boundaries and execution flow needed to add those pieces incrementally.

---

## Clone traffic

![Clone traffic](https://raw.githubusercontent.com/nikolareljin/stats/main/charts/finetorch.svg)

_Updated daily. Total and unique cloners over the last 14 days._
