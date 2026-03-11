# Use Cases

This document maps Finetorch to realistic workflows you are likely to run as the project matures.

## 1. Prepare Instruction-Tuning Data

Scenario:
- you have JSONL examples from multiple sources
- some use `instruction/input/output`
- some use `prompt/completion`

Why Finetorch fits:
- one CLI normalizes both formats
- deterministic train/val splitting
- shard generation for downstream training

Typical command:

```bash
finetorch prepare-dataset \
  --input data/train.jsonl \
  --output artifacts/dataset
```

## 2. Single-GPU LoRA Finetuning Workflow

Scenario:
- you want to finetune a model such as Gemma, Llama, or Mistral
- you want adapter training instead of full model retraining
- you need a config-driven workflow that stays reproducible

Why Finetorch fits:
- explicit run configuration
- backend abstraction for model runtime integration
- adapter-focused design
- local artifact layout for logs and outputs

Typical command:

```bash
finetorch train --config configs/example_run.toml
```

## 3. QLoRA-Oriented Small-Scale Experimentation

Scenario:
- you want lower-memory adapter runs on a single GPU
- you need to compare LoRA and QLoRA settings without changing your command flow

Why Finetorch fits:
- one config file can carry `use_qlora` and related run settings
- the backend layer can decide whether the chosen runtime supports the requested mode

Practical note:
- this depends on real backend support being implemented
- the current scaffold defines the configuration boundary but not the full kernel path yet

## 4. Evaluate a Finetune Before Export or Merge

Scenario:
- you have trained an adapter
- you want a quick quality pass before merging, exporting, or discarding the run

Why Finetorch fits:
- evaluation is a first-class CLI path
- metrics can be expanded over time
- train/eval artifacts stay in one project structure

Typical command:

```bash
finetorch eval \
  --config configs/example_run.toml \
  --dataset data/eval.jsonl
```

## 5. Prepare for GGUF Export and Quantized Inference Workflows

Scenario:
- you want to finetune first, then export for local inference
- you want smaller and cheaper-to-run artifacts for deployment

What Finetorch should become here:
- merge adapter into compatible weights
- orchestrate export to GGUF-compatible flows
- orchestrate quantization via trusted external tooling

Important distinction:
- quantization reduces precision and memory footprint
- quantization does not change the model architecture size
- a 7B model quantized to 4-bit is still a 7B model

## 6. Future Gemma Workflow

A realistic future workflow could be:
1. prepare dataset from instruction JSONL
2. run Gemma LoRA or QLoRA finetune
3. evaluate on held-out prompts
4. merge or keep adapters separately
5. export to an inference-friendly format
6. quantize for deployment

This is a good target use case for Finetorch because it combines:
- structured dataset prep
- config-driven training
- backend-specific model handling
- export and quantization orchestration

## 7. Distillation and True Model Compression

Scenario:
- you want a genuinely smaller-capacity student model, not just a quantized version of the same model

Important distinction:
- LoRA finetuning changes adapter weights
- quantization changes precision
- distillation changes the effective model architecture or training target

Finetorch can eventually support this, but it should be treated as a later roadmap feature with:
- teacher/student configs
- synthetic supervision workflows
- separate evaluation criteria

## 8. Why This Project Should Stay Orchestrator-First

Finetorch is strongest when it acts as:
- the workflow owner
- the config owner
- the artifact/layout owner
- the backend integration layer

It should avoid reinventing every low-level model conversion kernel itself. The practical direction is:
- use proven runtimes for model execution
- use proven converters/exporters where possible
- keep the Rust CLI and library responsible for orchestration, validation, and reproducibility
