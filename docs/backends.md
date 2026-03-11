# Backend and Adapter Design

Finetorch uses a backend abstraction to keep CLI and training orchestration independent from any single runtime.

## `LlmBackend` Trait

Defined in [src/model/backend.rs](../src/model/backend.rs).

Current interface:
- `load_model`
- `apply_lora`
- `forward`
- `save_adapters`

This is intentionally narrow. It is enough for:
- loading a model
- attaching LoRA state
- running a batch forward pass
- persisting adapter outputs

## Why a Backend Trait Exists

The toolkit needs to support multiple runtime families over time:
- `llama.cpp` style runtimes
- Rust-native tensor runtimes such as Candle
- future CUDA/ROCm-specific paths
- possible inference/training bridges to Python ecosystems

A trait boundary keeps those concerns contained.

## Current Backend: `llama_cpp`

The initial backend scaffold lives in [src/model/llama_cpp.rs](../src/model/llama_cpp.rs).

Current state:
- placeholder model load behavior
- placeholder LoRA application state
- synthetic forward output
- placeholder adapter file persistence

This is enough to validate module boundaries and CLI execution flow, but not enough yet for real gradient-based finetuning.

## LoRA and QLoRA Shape

The adapter configuration lives in [src/model/lora.rs](../src/model/lora.rs).

Current fields:
- `enabled`
- `use_qlora`
- `r`
- `alpha`
- `dropout`
- `target_modules`

These fields are deliberately backend-neutral. A backend should interpret them according to its capabilities.

## Expected Backend Responsibilities

A production backend implementation should own:
- loading model weights and tokenizer metadata
- quantization compatibility checks
- injection of LoRA layers into target modules
- forward pass execution on the configured device
- saving adapter weights in a stable format

A backend should not own:
- CLI parsing
- dataset splitting
- experiment naming
- metric aggregation logic

## Suggested Next Trait Extensions

As the project evolves, the backend trait will likely need:
- explicit training-step APIs
- gradient checkpointing capability reporting
- adapter merge/export operations
- quantization support flags
- tokenizer loading hooks
- device capability probing

Possible additions:

```rust
fn capabilities(&self) -> BackendCapabilities;
fn supports_qlora(&self) -> bool;
fn training_step(&mut self, batch: &ForwardBatch) -> Result<StepOutput>;
fn set_train_mode(&mut self) -> Result<()>;
fn set_eval_mode(&mut self) -> Result<()>;
```

## Integration Strategy for Real Backends

### `llama.cpp` bridge

Good fit for:
- GGUF model loading
- practical local model execution
- direct compatibility with common local inference environments

Risks:
- training support is less natural than pure tensor frameworks
- adapter training behavior may require custom bridge logic

### Candle backend

Good fit for:
- Rust-native tensor operations
- clearer training-loop integration
- better long-term control over adapter training code

Risks:
- more implementation work up front
- model compatibility and loading formats need deliberate handling

## Recommended Roadmap

1. Keep `llama_cpp` as the first compatibility bridge.
2. Add a Candle-based backend for more direct adapter training control.
3. Add tokenizer loading from model metadata.
4. Add capability-based config validation before runtime.
5. Add export bridges for adapter interchange.
