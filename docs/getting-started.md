# Getting Started

This guide shows the current practical way to use Finetorch as a local CLI scaffold for dataset preparation, config-driven training orchestration, and evaluation.

## What Finetorch Can Do Today

Current working scope:
- normalize supported JSONL training data
- shard datasets into train and validation splits
- load a TOML run configuration
- execute scaffolded train and eval flows
- emit adapter and metrics artifact placeholders

Current non-goals:
- production-grade GPU finetuning kernels
- full large-model pretraining
- native end-to-end quantization or distillation

## Prerequisites

- Rust toolchain installed
- a local clone of the repository

Build the project:

```bash
cd /home/dragana/Projects/finetorch
./scripts/build.sh
```

## Step 1: Create a Small Training Dataset

Finetorch accepts JSONL in either of these shapes:

```json
{"instruction":"Summarize the text","input":"...","output":"..."}
{"prompt":"Question: ...","completion":"Answer: ..."}
```

Example:

```bash
mkdir -p data
cat > data/train.jsonl <<'EOF'
{"instruction":"Answer briefly","input":"What is LoRA?","output":"LoRA is a parameter-efficient finetuning method."}
{"prompt":"Complete: Gemma is","completion":"a family of language models."}
EOF
```

## Step 2: Prepare the Dataset

Run:

```bash
cargo run -- prepare-dataset \
  --input data/train.jsonl \
  --output artifacts/dataset \
  --tokenizer whitespace \
  --train-ratio 0.95 \
  --shard-size 1024
```

This produces:

```text
artifacts/dataset/
  manifest.json
  train/
    shard-00000.jsonl
  val/
    shard-00000.jsonl
```

## Step 3: Inspect the Example Run Config

The scaffolded config lives in [../configs/example_run.toml](../configs/example_run.toml).

Key sections:
- `[model]`
- `[lora]`
- `[training]`
- `[data]`
- `[evaluation]`

If you want a smoke-test run, keep:
- low `max_steps`
- small `batch_size`
- one local output directory per run

## Step 4: Run Training

Run:

```bash
cargo run -- train --config configs/example_run.toml
```

Current scaffold behavior:
- builds the configured backend
- loads the configured model path logically
- applies LoRA settings logically
- writes training logs
- saves placeholder adapter output

Expected artifacts:

```text
artifacts/run-001/
  adapter.safetensors
  train_metrics.jsonl
```

## Step 5: Run Evaluation

Use either a separate eval file or a small held-out JSONL.

Example:

```bash
cat > data/eval.jsonl <<'EOF'
{"prompt":"What is QLoRA?","completion":"A quantized LoRA finetuning approach."}
EOF
```

Then run:

```bash
cargo run -- eval \
  --config configs/example_run.toml \
  --dataset data/eval.jsonl
```

The current scaffold prints summary metrics such as:
- perplexity
- exact match
- BLEU
- ROUGE-L

## Local Validation Commands

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

Run the full local CI sequence:

```bash
./scripts/local_ci.sh
```

## Recommended Directory Layout

```text
finetorch/
  configs/
    example_run.toml
  data/
    train.jsonl
    eval.jsonl
  models/
    llama-3-8b/
  artifacts/
    dataset/
    run-001/
```

## What To Do Next

Once the scaffold is working locally, the most practical next steps are:
1. replace placeholder tokenization with a real tokenizer backend
2. replace synthetic training batches with prepared dataset loading
3. deepen the `llama_cpp` backend into a real runtime bridge
4. add export and quantization orchestration
