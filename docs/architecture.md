# Architecture

Finetorch is organized as a CLI-first Rust application with reusable library modules behind it. The design goal is to keep dataset handling, model backend integration, and training orchestration separate so new backends and tasks can be added without rewriting the command layer.

## System Layout

Finetorch is split into five logical areas:

1. CLI
   - Entry point: `src/main.rs`
   - Command routing: `src/cli/`
   - Responsibilities:
     - parse user intent
     - validate inputs and config paths
     - call data, training, and evaluation workflows

2. Configuration
   - Source: `src/config.rs`
   - Responsibilities:
     - deserialize `run.toml`
     - keep runtime configuration strongly typed
     - define stable configuration boundaries between modules

3. Data Pipeline
   - Source: `src/data/`
   - Responsibilities:
     - ingest JSONL training examples
     - normalize schema variants into one internal representation
     - tokenize examples using a selected tokenizer strategy
     - split and shard data for training and validation

4. Model Backend Layer
   - Source: `src/model/`
   - Responsibilities:
     - expose a backend-neutral `LlmBackend` trait
     - encapsulate adapter application and saving
     - isolate backend-specific loading and forward-pass behavior
     - provide a controlled extension point for future runtimes

5. Training and Evaluation
   - Sources: `src/train/`, `src/eval/`
   - Responsibilities:
     - orchestrate training runs from config
     - hold optimizer and scheduler state definitions
     - emit adapter outputs and training logs
     - compute evaluation summaries and task metrics

## End-to-End Data Flow

### Dataset Preparation

`finetorch prepare-dataset --input data.jsonl --output dataset/`

Flow:
1. Read JSONL records from disk.
2. Normalize records into `TrainingExample { prompt, completion }`.
3. Select tokenizer behavior from CLI input.
4. Tokenize examples and record token counts.
5. Shuffle deterministically.
6. Split into train and validation partitions.
7. Write sharded JSONL files plus a manifest.

Outputs:
- `dataset/train/shard-*.jsonl`
- `dataset/val/shard-*.jsonl`
- `dataset/manifest.json`

### Finetuning Run

`finetorch train --config configs/example_run.toml`

Flow:
1. Load `RunConfig` from TOML.
2. Build the selected backend implementation.
3. Load the base model from the configured path.
4. Apply LoRA or QLoRA configuration.
5. Construct optimizer and scheduler settings.
6. Execute the training loop.
7. Emit adapter weights and training metrics.

Outputs:
- adapter file under `training.output_dir`
- JSONL logs under `training.output_dir`

### Evaluation Run

`finetorch eval --config configs/example_run.toml --dataset eval.jsonl`

Flow:
1. Load run config.
2. Load evaluation examples.
3. Build and initialize the configured backend.
4. Run evaluation over normalized records.
5. Aggregate metric summaries.

Outputs:
- CLI summary of perplexity, exact match, BLEU, and ROUGE-L

## Module Boundaries

### `src/cli/`

The CLI layer should remain thin. It should not contain tokenizer logic, dataset sharding rules, or backend-specific training code. Its role is orchestration and presentation.

### `src/data/`

The data layer owns example normalization and sharding. This keeps input-format complexity out of the backend and training loop.

### `src/model/`

The model layer is the extension seam. If Finetorch gains a Candle, tch-rs, or custom CUDA backend later, the implementation should live here while preserving the `LlmBackend` interface.

### `src/train/`

The training layer should stay backend-agnostic. It should reason in terms of batches, optimizer state, scheduler state, and artifacts, not runtime-specific tensor APIs.

### `src/eval/`

The evaluation layer should remain metric-focused and task-oriented. Metrics should be additive and easy to disable or expand.

## Extension Strategy

Recommended next additions:

1. Replace placeholder tokenization with a real tokenizer bridge.
2. Replace synthetic training batches with dataset-driven iterators.
3. Expand `LlmBackend` to support gradient steps or adapter-specific update hooks.
4. Add metric-specific evaluation implementations per task type.
5. Add backend capability reporting so unsupported combinations fail early.

## Single-GPU Design Assumption

Finetorch is intentionally optimized for practical, single-node workflows:
- one GPU or one active accelerator target per run
- adapter training rather than full model pretraining
- explicit config for batch size and gradient accumulation
- small, inspectable training artifacts

That constraint keeps the architecture simple and makes the CLI and config model stable while the backend layer evolves.
