# Configuration Guide

Finetorch uses TOML-based run configuration files. The canonical example is [configs/example_run.toml](../configs/example_run.toml).

## Top-Level Sections

A typical run file contains:
- `[model]`
- `[lora]`
- `[training]`
- `[data]`
- `[evaluation]`

## `[model]`

Example:

```toml
[model]
base_path = "models/llama-3-8b"
backend = "llama_cpp"
tokenizer = "sentencepiece:models/llama-3/tokenizer.model"
```

Fields:
- `base_path`
  - Filesystem path to the base model or model directory.
- `backend`
  - Backend implementation name.
  - Current scaffolded backend: `llama_cpp`.
- `tokenizer`
  - Tokenizer selection string.
  - Current scaffold accepts the value but does not yet map named tokenizers to production tokenization kernels.

## `[lora]`

Example:

```toml
[lora]
enabled = true
use_qlora = true
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

Fields:
- `enabled`
  - Enables adapter application.
- `use_qlora`
  - Indicates 4-bit/quantized adapter workflow intent.
- `r`
  - LoRA rank.
- `alpha`
  - LoRA scaling value.
- `dropout`
  - Adapter dropout ratio.
- `target_modules`
  - List of module names expected to receive adapter layers.

## `[training]`

Example:

```toml
[training]
device = "cuda:0"
batch_size = 8
gradient_accumulation = 4
lr = 0.0002
epochs = 3
max_steps = 1000
warmup_steps = 50
weight_decay = 0.01
seed = 42
output_dir = "artifacts/run-001"
```

Fields:
- `device`
  - Execution target, such as `cuda:0`, `rocm:0`, `cpu`, or a future `wgpu` device string.
- `batch_size`
  - Per-step micro-batch size.
- `gradient_accumulation`
  - Number of micro-batches accumulated before an optimizer step.
- `lr`
  - Learning rate.
- `epochs`
  - Number of passes over the prepared dataset.
- `max_steps`
  - Upper bound for training steps.
- `warmup_steps`
  - Warmup length for scheduler behavior.
- `weight_decay`
  - Optimizer regularization value.
- `seed`
  - Reproducibility seed.
- `output_dir`
  - Target directory for logs and adapters.

## `[data]`

Example:

```toml
[data]
train_path = "artifacts/dataset/train"
val_path = "artifacts/dataset/val"
max_seq_len = 2048
shuffle = true
```

Fields:
- `train_path`
  - Directory or manifest path for training shards.
- `val_path`
  - Directory or manifest path for validation shards.
- `max_seq_len`
  - Maximum token length per example or packed sequence.
- `shuffle`
  - Whether to shuffle records before batching.

## `[evaluation]`

Example:

```toml
[evaluation]
metrics = ["perplexity", "exact_match", "bleu", "rouge_l"]
```

Fields:
- `metrics`
  - Ordered list of metric names to compute.

## Recommended Small-Scale Profiles

### Small 8B Adapter Run

Use this when experimenting on one consumer GPU:
- `r = 8` or `16`
- `batch_size = 2` to `8`
- `gradient_accumulation = 4` to `16`
- `max_seq_len = 1024` or `2048`
- `use_qlora = true` if backend supports it

### Quick Smoke Test

Use this for first-run validation:
- `epochs = 1`
- `max_steps = 10`
- `batch_size = 1`
- `gradient_accumulation = 1`
- a tiny dataset shard set

## Validation Rules to Add Later

The current scaffold accepts structurally valid TOML. Production validation should later enforce:
- non-empty `model.base_path`
- supported backend names
- positive LoRA rank
- `0.0 <= dropout < 1.0`
- positive batch and accumulation sizes
- valid train/val dataset paths
- metric names from a supported set

## Operational Guidance

Keep one config file per run family, for example:
- `configs/llama3_smoke.toml`
- `configs/llama3_qlora_8b.toml`
- `configs/mistral_eval_only.toml`

That keeps the CLI simple and makes runs reproducible in CI or local automation.
