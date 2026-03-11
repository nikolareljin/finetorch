# CLI Workflows

This document shows the expected command flow for common Finetorch tasks.

## 1. Prepare a Dataset

Finetorch accepts JSONL in either of these shapes:

```json
{"instruction":"Summarize the text","input":"...","output":"..."}
{"prompt":"Question: ...","completion":"Answer: ..."}
```

Example command:

```bash
finetorch prepare-dataset \
  --input data/train.jsonl \
  --output artifacts/dataset \
  --tokenizer sentencepiece:models/llama-3/tokenizer.model \
  --train-ratio 0.95 \
  --shard-size 2048
```

Expected output layout:

```text
artifacts/dataset/
  manifest.json
  train/
    shard-00000.jsonl
    shard-00001.jsonl
  val/
    shard-00000.jsonl
```

Current scaffold behavior:
- normalizes the records
- performs deterministic shuffle/split
- records token counts using the current tokenizer strategy

## 2. Start a Training Run

Example command:

```bash
finetorch train --config configs/example_run.toml
```

What happens:
1. The config is loaded.
2. The backend is selected.
3. The base model is loaded.
4. LoRA configuration is applied.
5. The training loop writes JSONL metrics.
6. Adapter weights are saved to the configured output directory.

Expected artifacts:

```text
artifacts/run-001/
  adapter.safetensors
  train_metrics.jsonl
```

## 3. Evaluate a Dataset

Example command:

```bash
finetorch eval \
  --config configs/example_run.toml \
  --dataset data/eval.jsonl
```

Current scaffold output includes:
- example count
- perplexity
- exact match
- BLEU
- ROUGE-L

## 4. Local Development Workflow

Build:

```bash
./scripts/build.sh
```

Test:

```bash
./scripts/test.sh
```

Lint:

```bash
./scripts/lint.sh
```

Full local CI pass:

```bash
./scripts/local_ci.sh
```

## 5. Common Iteration Pattern

A practical single-GPU workflow usually looks like this:

1. Clean and inspect the raw JSONL.
2. Prepare shards with `prepare-dataset`.
3. Start with a smoke-test config using very low `max_steps`.
4. Inspect `train_metrics.jsonl` for loss stability.
5. Run `eval` on a held-out set.
6. Increase rank, steps, or sequence length only after the smoke run is stable.

## 6. Failure Modes to Expect

While the scaffold is intentionally lightweight, these are the first operational checks to add as implementation deepens:
- missing model files
- unsupported backend names
- invalid tokenizer references
- non-existent dataset directories
- invalid LoRA settings
- GPU device mismatch or unsupported quantization mode

## 7. Recommended Repository Conventions

Suggested directories for day-to-day usage:

```text
models/
data/
artifacts/
configs/
```

Example:

```text
models/llama-3-8b/
data/train.jsonl
data/eval.jsonl
artifacts/dataset/
artifacts/run-001/
configs/example_run.toml
```
